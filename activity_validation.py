"""Independent-rating evaluation of frozen descriptive clusters; no reminder activation."""
import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from activity_clustering import FEATURE_VERSION, iso_date, safe_read

NUMERIC = ["hour_sin", "hour_cos", "prior_known_work_minutes", "prior_labels_unavailable"]


def read_study_ratings(source, manifest, schedule_version, allow_non_synthetic=False):
    mapping = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if not {"session_id", "participant_id"}.issubset(mapping.columns) or mapping.session_id.duplicated().any() or (mapping.participant_id == "").any():
        raise ValueError("Independent validation needs unique sessions and stable participant pseudonyms")
    identities = mapping.set_index("session_id").participant_id.to_dict()
    records = []
    source = Path(source)
    folders = [source] if (source / "session.json").is_file() else sorted(path.parent for path in source.glob("*/session.json"))
    for folder in folders:
        if folder.is_symlink():
            raise ValueError("Unsafe recording folder")
        metadata = json.loads(safe_read(folder, "session.json").read_text())
        if metadata.get("schema_version") != 11 or (not allow_non_synthetic and metadata.get("synthetic_fixture") is not True):
            raise ValueError("Independent validation defaults to generated schema-11 recordings")
        session_id = metadata.get("session_id", folder.name)
        if session_id not in identities:
            raise ValueError("Missing stable participant pseudonym")
        prompts = {}
        log = folder / "study_check_ins.jsonl"
        if not log.exists():
            continue
        for line in safe_read(folder, log.name).read_text().splitlines():
            entry = json.loads(line)
            prompt = entry["prompt"]
            if entry["action"] == "requested" and prompt["scheduleVersion"] == schedule_version:
                if metadata.get("synthetic_fixture") is not True and prompt.get("configurationScope") != "reviewed_study":
                    raise ValueError("Real validation requires the independently reviewed study scope")
                if prompt["id"] in prompts:
                    raise ValueError("Duplicate independent prompt request")
                prompts[prompt["id"]] = prompt
        surveys = pd.read_csv(safe_read(folder, "surveys.csv"), keep_default_na=False)
        if surveys.check_in_id.duplicated().any():
            raise ValueError("Duplicate survey answer")
        for identifier, prompt in prompts.items():
            answer = surveys.loc[surveys.check_in_id == identifier]
            row = answer.iloc[0] if len(answer) else None
            status = str(row.status) if row is not None else "missing"
            rating = float(row.fatigue_0_10) if row is not None and row.fatigue_0_10 != "" else np.nan
            if status != "answered" and np.isfinite(rating):
                raise ValueError("Missing/skipped ratings must stay missing")
            if np.isfinite(rating) and not (0 <= rating <= 10 and rating.is_integer()):
                raise ValueError("Invalid independent fatigue rating")
            records.append(dict(check_in_id=identifier, participant_id=identities[session_id], session_id=session_id,
                status=status, fatigue=rating, context=prompt["contextAtPrompt"], request_elapsed=float(prompt["requestedElapsed"]),
                recall_start=float(prompt["recallStart"]), recall_end=float(prompt["recallEnd"]),
                request_utc=pd.Timestamp(float(prompt["requestedAtUTC"]), unit="s", tz="UTC").isoformat(),
                answered_elapsed=float(row.answered_elapsed_seconds) if row is not None else np.nan,
                schedule_version=schedule_version, independent=True, synthetic=metadata.get("synthetic_fixture") is True))
    return pd.DataFrame(records)


def join_ratings(assignments, ratings):
    rows, audit = [], []
    if ratings.empty:
        return pd.DataFrame(), [dict(reason="no_independent_ratings")]
    if ratings.check_in_id.duplicated().any():
        raise ValueError("Duplicate check-in ID")
    windows = assignments.copy()
    if windows.window_id.duplicated().any() or not windows.split.isin(["development", "test"]).all():
        raise ValueError("Duplicate windows or unsupported split assignments")
    windows["end_utc"] = pd.to_datetime(windows.start_utc, utc=True) + pd.to_timedelta(windows.end_seconds - windows.start_seconds, unit="s")
    for rating in ratings.sort_values(["request_utc", "check_in_id"]).itertuples():
        reason = None
        if not rating.independent:
            reason = "not_independent"
        elif rating.status != "answered" or not np.isfinite(rating.fatigue):
            reason = "missing_or_skipped"
        elif rating.context not in {"work", "non_work", "unknown"}:
            reason = "invalid_prompt_context"
        elif not (np.isfinite(rating.recall_start) and np.isfinite(rating.recall_end) and 0 <= rating.recall_start < rating.recall_end <= rating.request_elapsed
                  and rating.answered_elapsed >= rating.request_elapsed and 0 <= rating.fatigue <= 10):
            reason = "invalid_recall_or_rating"
        candidates = windows.loc[(windows.participant_id == rating.participant_id) & (windows.session_id == rating.session_id)
            & (windows.start_seconds >= rating.recall_start - 1e-6) & (windows.end_seconds <= rating.recall_end + 1e-6)
            & (windows.end_seconds <= rating.request_elapsed)] if not reason else pd.DataFrame()
        if not reason and candidates.empty:
            reason = "no_fully_preceding_covered_window"
        if reason:
            audit.append(dict(check_in_id=rating.check_in_id, context=rating.context, reason=reason)); continue
        window = candidates.sort_values("end_seconds").iloc[-1]
        request_date = pd.Timestamp(rating.request_utc)
        if request_date.tzinfo is None:
            raise ValueError("Prompt UTC needs timezone")
        if window.end_utc > request_date:
            audit.append(dict(check_in_id=rating.check_in_id, context=rating.context, reason="window_after_request")); continue
        previous = windows.loc[(windows.participant_id == rating.participant_id) & (windows.end_utc <= request_date)
            & (windows.end_utc.dt.date == request_date.date())]
        known = previous.loc[previous.labels_recorded_at_utc <= request_date.timestamp()] if "labels_recorded_at_utc" in previous else previous.iloc[:0]
        prior = float(((known.end_seconds - known.start_seconds) * known.work_fraction).sum() / 60)
        hour = request_date.hour + request_date.minute / 60
        rows.append(dict(check_in_id=rating.check_in_id, participant_id=rating.participant_id, request_utc=rating.request_utc,
            day=request_date.date().isoformat(), split=window.split, cluster_id=str(window.cluster_id), context=rating.context,
            fatigue=float(rating.fatigue), answered_utc=(request_date + pd.to_timedelta(rating.answered_elapsed - rating.request_elapsed, unit="s")).isoformat(),
            window_id=window.window_id, window_end_seconds=window.end_seconds,
            recall_end=rating.recall_end, request_elapsed=rating.request_elapsed, hour_sin=float(np.sin(2 * np.pi * hour / 24)),
            hour_cos=float(np.cos(2 * np.pi * hour / 24)), prior_known_work_minutes=prior,
            prior_labels_unavailable=int(len(previous) - len(known))))
        audit.append(dict(check_in_id=rating.check_in_id, context=rating.context, reason="included"))
    return pd.DataFrame(rows), audit


def earlier_personal_ratings(table):
    result = table.copy()
    # Zero is a missing-feature placeholder paired with an explicit availability flag,
    # never an invented fatigue answer. All real personal history is strictly earlier.
    result["prior_fatigue"] = 0.0; result["prior_rating_available"] = 0.0
    for _, group in result.groupby("participant_id"):
        for index, item in group.iterrows():
            prior = group.loc[(pd.to_datetime(group.request_utc, utc=True) < pd.Timestamp(item.request_utc))
                & (pd.to_datetime(group.answered_utc, utc=True) < pd.Timestamp(item.request_utc)), "fatigue"]
            if len(prior):
                result.loc[index, "prior_fatigue"] = float(prior.mean())
                result.loc[index, "prior_rating_available"] = 1.0
    return result


def evaluate(assignments, ratings, cluster_report, output, mode="new-person"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    if cluster_report.get("feature_version") != FEATURE_VERSION or cluster_report.get("status") != "technical_fit_only":
        raise ValueError("A frozen compatible clustering fit is required")
    if cluster_report.get("mode") != ("unseen" if mode == "new-person" else "personal"):
        raise ValueError("Cluster fitting split does not match evaluation generalization")
    table, audit = join_ratings(assignments, ratings)
    missingness = ratings.groupby(["context", "status"]).size().to_dict() if not ratings.empty else {}
    report = dict(status="insufficient_evidence", mode=mode, synthetic=bool(cluster_report.get("synthetic")),
        model_version=cluster_report.get("model_version"), requested=len(ratings), eligible=len(table),
        missingness=[dict(context=context, status=status, count=int(count)) for (context, status), count in missingness.items()],
        cluster_structure=dict(development_silhouette=cluster_report.get("development_silhouette"), bootstrap_ari=cluster_report.get("bootstrap_development_ari"), real_world_stability="not_established"),
        fatigue_prediction="not_established", intervention_benefit="not_tested", productivity="not_evaluated; input and declared minutes are not quality-adjusted output")
    (output / "joining.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not table.empty:
        table.to_csv(output / "eligible_ratings.csv", index=False)
        dev = table.loc[table.split == "development"]
        test = table.loc[table.split == "test"]
        report.update(development_ratings=len(dev), test_ratings=len(test), development_people=int(dev.participant_id.nunique()), test_people=int(test.participant_id.nunique()))
        enough = len(dev) >= 30 and len(test) >= 20
        if mode == "new-person":
            if set(dev.participant_id) & set(test.participant_id):
                raise ValueError("Participant leakage")
            enough &= dev.participant_id.nunique() >= 3 and test.participant_id.nunique() >= 3
        else:
            if table.participant_id.nunique() != 1 or dev.empty or test.empty or max(pd.to_datetime(dev.request_utc)) >= min(pd.to_datetime(test.request_utc)):
                raise ValueError("Later-time calibration must use only earlier observations from one person")
            table = earlier_personal_ratings(table)
            dev, test = table.loc[table.split == "development"], table.loc[table.split == "test"]
        if enough:
            numeric = NUMERIC + (["prior_fatigue", "prior_rating_available"] if mode == "later-time" else [])
            predictions = test[["check_in_id", "participant_id", "request_utc", "fatigue"]].copy()
            predictions["training_mean"] = float(dev.fatigue.mean())
            for name, categorical in [("context_time_prior_work", ["context"]), ("plus_cluster", ["context", "cluster_id"])]:
                preprocessing = ColumnTransformer([("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
                model = make_pipeline(preprocessing, Ridge(alpha=1.0))
                model.fit(dev, dev.fatigue)
                predictions[name] = np.clip(model.predict(test), 0, 10)
            errors = {name: np.abs(predictions[name] - predictions.fatigue) for name in ["training_mean", "context_time_prior_work", "plus_cluster"]}
            report["held_out_mae"] = {name: float(values.mean()) for name, values in errors.items()}
            difference = errors["context_time_prior_work"] - errors["plus_cluster"]
            if mode == "new-person":
                by_person = difference.groupby(predictions.participant_id).mean().to_numpy()
                rng = np.random.default_rng(43)
                draws = [float(rng.choice(by_person, size=len(by_person), replace=True).mean()) for _ in range(1000)]
                report["participant_bootstrap_mae_improvement_interval"] = np.quantile(draws, [.025, .975]).tolist()
            else:
                report["uncertainty"] = "One-person/later-day technical comparison; participant resampling is unavailable, so generalization remains uncertain"
            report.update(status="synthetic_technical_comparison" if report["synthetic"] else "descriptive_technical_comparison",
                fatigue_prediction="Synthetic feasibility only" if report["synthetic"] else "Descriptive held-out errors; adequacy/power/generalization require independent study review",
                selection="Ridge alpha=1 fixed in advance; no candidate selected using final-test results",
                statistical_adequacy="Technical sample gates are not proof of adequate statistical evidence")
            predictions.to_csv(output / "held_out_predictions.csv", index=False)
        else:
            report["reason"] = "Predeclared technical count/participant gates not met; no accuracy claim"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def synthetic_ratings(assignments):
    rng = np.random.default_rng(47)
    result = []
    for index, row in enumerate(assignments.itertuples()):
        requested = iso_date(row.start_utc) + timedelta(seconds=row.end_seconds - row.start_seconds)
        result.append(dict(check_in_id=f"synthetic-rating-{index}", participant_id=row.participant_id, session_id=row.session_id,
            status="answered" if index % 7 else "skipped", fatigue=float(rng.integers(0, 11)) if index % 7 else np.nan,
            context=row.context if row.context != "mixed" else "unknown", request_elapsed=row.end_seconds,
            recall_start=row.start_seconds, recall_end=row.end_seconds, request_utc=requested.isoformat(),
            answered_elapsed=row.end_seconds + 20, schedule_version="synthetic.demo.v1", independent=True, synthetic=True))
    return pd.DataFrame(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "evaluate"]:
        command = commands.add_parser(name); command.add_argument("--cluster-run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--mode", choices=["new-person", "later-time"], default="new-person")
    actual = commands.choices["evaluate"]
    actual.add_argument("--recordings", type=Path, required=True); actual.add_argument("--manifest", type=Path, required=True)
    actual.add_argument("--schedule-version", required=True); actual.add_argument("--allow-non-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        assignments = pd.read_csv(args.cluster_run / "assignments.csv")
        cluster_report = json.loads((args.cluster_run / "report.json").read_text())
        if args.command == "demo":
            if not cluster_report.get("synthetic"):
                raise ValueError("Demo must use synthetic clustering artifacts")
            ratings = synthetic_ratings(assignments)
        else:
            ratings = read_study_ratings(args.recordings, args.manifest, args.schedule_version, args.allow_non_synthetic)
        print(json.dumps(evaluate(assignments, ratings, cluster_report, args.output, args.mode), indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
