"""Separate schema-11 activity clustering. Generated data by default; no service access."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_VERSION = "focusfix_activity_rates_v1"
FORMAT = "focusfix.recording.v11"
CAPABILITIES = ["context_intervals_v1", "independent_intervention_phase", "monotonic_elapsed_v1"]
FEATURES = ["keys_per_minute", "deletes_per_minute", "clicks_per_minute", "switches_per_minute",
            "scrolls_per_minute", "mouse_points_per_minute", "mouse_events_per_minute"]
WINDOW_SECONDS = 300
CANDIDATE_K = (2, 3, 4)
PHASES = {"work", "check_in", "awaiting_return"}  # Phase only, never a work-context label.
CONTEXTS = {"work", "non_work", "unknown"}


def safe_read(folder, name):
    path = folder / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError(f"Missing, unsafe or oversized {name}")
    return path


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    return float(value)


def iso_date(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("UTC date needs an explicit timezone")
    return result.astimezone(timezone.utc)


def load_recording(folder, participant, allow_non_synthetic=False):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Unsafe recording folder")
    metadata = json.loads(safe_read(folder, "session.json").read_text())
    if metadata.get("schema_version") != 11 or metadata.get("recording_format") != FORMAT or metadata.get("required_capabilities") != CAPABILITIES:
        raise ValueError("Unsupported schema or required capabilities")
    if not allow_non_synthetic and metadata.get("synthetic_fixture") is not True:
        raise ValueError("Generated fixtures only unless explicitly enabled for a separate approved analysis")
    contract = json.loads(safe_read(folder, "context_intervals.json").read_text())
    if contract.get("format") != FORMAT or contract.get("finalized") is not True:
        raise ValueError("A finalized context contract is required")
    for metadata_key, contract_key in (("recording_id", "recordingID"), ("segment_id", "segmentID")):
        UUID(contract[contract_key])
        if metadata.get(metadata_key) != contract[contract_key]:
            raise ValueError("Recording/segment identity mismatch")
    if metadata.get("session_id") != folder.name or metadata.get("recording_mode") != contract.get("mode") or contract.get("mode") not in {"manual", "automatic"}:
        raise ValueError("Session/mode identity mismatch")
    origin, end = finite(contract["elapsedOrigin"], "origin"), finite(contract["observedThrough"], "end")
    if origin < 0 or end <= origin or abs(finite(metadata.get("duration_seconds"), "duration") - (end - origin)) > 1e-6:
        raise ValueError("Invalid recording bounds")
    previous, ids = origin, set()
    for label in contract["intervals"]:
        UUID(label["id"]); UUID(label["logicalID"])
        lower, upper = finite(label["start"], "label start"), finite(label["end"], "label end")
        if label["id"] in ids or lower < previous or upper <= lower or upper > end or label["context"] not in CONTEXTS:
            raise ValueError("Overlapping or malformed context interval")
        finite(label["effectiveAtUTC"], "effective UTC"); finite(label["recordedAtUTC"], "recorded UTC")
        if label["source"] not in {"manual_start", "automatic_default", "user_action", "history_edit", "notification", "lifecycle"}:
            raise ValueError("Unsupported label provenance")
        ids.add(label["id"]); previous = upper
    revisions = contract.get("revisions", [])
    if [r["sequence"] for r in revisions] != list(range(1, len(revisions) + 1)) or len({r["id"] for r in revisions}) != len(revisions):
        raise ValueError("Invalid revision ordering")
    events = pd.read_csv(safe_read(folder, "events.csv"), keep_default_na=False)
    mouse = pd.read_csv(safe_read(folder, "mouse_movement.csv"), keep_default_na=False)
    for table, columns in ((events, ["elapsed_seconds"]), (mouse, ["start_seconds", "end_seconds", "distance_screen_points", "movement_events"])):
        if not set(columns + ["activity_phase"]).issubset(table.columns):
            raise ValueError("Missing activity columns")
        for column in columns:
            table[column] = pd.to_numeric(table[column], errors="raise")
            if not np.isfinite(table[column]).all() or (table[column] < 0).any():
                raise ValueError("Invalid activity values")
        if not table.activity_phase.isin(PHASES).all():
            raise ValueError("Unknown intervention phase")
    if not {"event_type"}.issubset(events.columns):
        raise ValueError("Missing event type")
    if (mouse.movement_events % 1 != 0).any():
        raise ValueError("Noninteger movement count")
    mouse = mouse.sort_values("start_seconds")
    if (mouse.end_seconds <= mouse.start_seconds).any() or (mouse.start_seconds.to_numpy()[1:] < mouse.end_seconds.to_numpy()[:-1] - 1e-6).any():
        raise ValueError("Overlapping mouse observations")
    recovered = metadata.get("status") == "crash_recovered_observed_only"
    if not recovered and ((mouse.end_seconds > end + 1e-6).any() or (events.elapsed_seconds > end + 1e-6).any()):
        raise ValueError("Unbounded raw activity tail")
    # Uncheckpointed crash tails cannot enter features, labels, or evaluation.
    events = events.loc[(events.elapsed_seconds >= origin) & (events.elapsed_seconds < end)].copy()
    mouse = mouse.loc[(mouse.start_seconds >= origin) & (mouse.end_seconds <= end + 1e-6)].copy()
    breaks = []
    with safe_read(folder, "breaks.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    starts, finishes = {}, {}
    for row in records:
        if row.get("break_id") and row.get("start_seconds"):
            starts[row["break_id"]] = float(row["start_seconds"])
        if row.get("action") == "break_end" and row.get("end_seconds"):
            finishes[row["break_id"]] = float(row["end_seconds"])
    for key, lower in starts.items():
        upper = finishes.get(key, end)
        if not math.isfinite(lower + upper) or lower < 0 or upper < lower:
            raise ValueError("Invalid break bounds")
        breaks.append((max(origin, lower), min(end, upper)))
    started = iso_date(metadata["started_at_utc"])
    zone = ZoneInfo(contract.get("scoringTimeZone", "UTC"))
    return metadata, contract, events, mouse, breaks, started, zone


def rate_features(events, mouse, lower, upper):
    """Full-coverage normal-phase windows only; float32 rates in fixed order."""
    bins = mouse.loc[(mouse.start_seconds >= lower - 1e-6) & (mouse.end_seconds <= upper + 1e-6)
                     & (mouse.activity_phase == "work")]
    observed = float((bins.end_seconds - bins.start_seconds).sum())
    if abs(observed - (upper - lower)) > 1e-5 or not len(bins):
        return None
    activity = events.loc[(events.elapsed_seconds >= lower) & (events.elapsed_seconds < upper) & (events.activity_phase == "work")]
    types = activity.event_type
    factor = 60.0 / observed
    values = [types.isin(["key_down", "delete_down"]).sum(), (types == "delete_down").sum(),
              types.isin(["left_click_down", "right_click_down", "other_click_down"]).sum(),
              (types == "app_switch").sum(), (types == "scroll").sum(),
              bins.distance_screen_points.sum(), bins.movement_events.sum()]
    return dict(zip(FEATURES, [float(np.float32(float(value) * factor)) for value in values]))


def prepare_windows(source, manifest, allow_non_synthetic=False):
    mapping = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if not {"session_id", "participant_id"}.issubset(mapping.columns) or mapping.session_id.duplicated().any() or (mapping.participant_id == "").any():
        raise ValueError("Manifest requires unique sessions and stable nonempty participant pseudonyms")
    source = Path(source)
    folders = [source] if (source / "session.json").is_file() else sorted(path.parent for path in source.glob("*/session.json"))
    identities = dict(zip(mapping.session_id, mapping.participant_id))
    rows, audit = [], []
    seen_segments = set()
    for folder in folders:
        if folder.name not in identities:
            raise ValueError("Every recording needs a stable manifest identity")
        metadata, contract, events, mouse, breaks, started, zone = load_recording(folder, identities[folder.name], allow_non_synthetic)
        if contract["segmentID"] in seen_segments:
            raise ValueError("Duplicate segment cannot duplicate windows")
        seen_segments.add(contract["segmentID"])
        origin, end = contract["elapsedOrigin"], contract["observedThrough"]
        checkins, open_checkin = [], None
        for item in events.sort_values("elapsed_seconds").itertuples():
            if item.event_type == "check_in_start":
                if open_checkin is not None:
                    raise ValueError("Overlapping check-in interaction")
                open_checkin = float(item.elapsed_seconds)
            elif item.event_type == "check_in_end" and open_checkin is not None:
                checkins.append((open_checkin, float(item.elapsed_seconds))); open_checkin = None
        if open_checkin is not None:
            checkins.append((open_checkin, end))
        lower = math.ceil(origin / WINDOW_SECONDS) * WINDOW_SECONDS
        if lower > origin:
            audit.append(dict(session_id=folder.name, reason="partial_boundary", start=origin, end=lower))
        while lower + WINDOW_SECONDS <= end + 1e-6:
            upper = lower + WINDOW_SECONDS
            reason = None
            if any(max(lower, a) < min(upper, b) for a, b in breaks):
                reason = "explicit_break"
            elif any(max(lower, a) < min(upper, b) for a, b in checkins) or ((mouse.start_seconds < upper) & (mouse.end_seconds > lower) & (mouse.activity_phase != "work")).any():
                reason = "check_in_or_awaiting_return"
            features = None if reason else rate_features(events, mouse, lower, upper)
            if features is None:
                audit.append(dict(session_id=folder.name, reason=reason or "insufficient_observed_coverage", start=lower, end=upper))
            else:
                fractions = {context: 0.0 for context in CONTEXTS}
                for label in contract["intervals"]:
                    fractions[label["context"]] += max(0, min(upper, label["end"]) - max(lower, label["start"])) / WINDOW_SECONDS
                fractions["unknown"] += max(0, 1 - sum(fractions.values()))
                context = next((key for key, value in fractions.items() if value >= 1 - 1e-6), "mixed")
                utc = started + timedelta(seconds=lower - origin)
                rows.append(dict(window_id=f"{contract['segmentID']}:{lower:g}", session_id=folder.name,
                    recording_id=contract["recordingID"], segment_id=contract["segmentID"], participant_id=identities[folder.name],
                    start_seconds=lower, end_seconds=upper, start_utc=utc.isoformat(), day=utc.astimezone(zone).date().isoformat(),
                    context=context, work_fraction=fractions["work"], non_work_fraction=fractions["non_work"], unknown_fraction=fractions["unknown"],
                    synthetic=metadata.get("synthetic_fixture") is True, coverage_fraction=1.0, **features))
                audit.append(dict(session_id=folder.name, reason="included", start=lower, end=upper))
            lower = upper
        if lower < end:
            audit.append(dict(session_id=folder.name, reason="partial_boundary", start=lower, end=end))
    if not folders:
        raise ValueError("No schema-11 recording folders found")
    return pd.DataFrame(rows), audit


def assign_split(table, mode):
    if table.empty:
        raise ValueError("Insufficient observed windows")
    participants = sorted(table.participant_id.unique())
    if mode == "unseen":
        if len(participants) < 4:
            raise ValueError("Unseen-person clustering requires at least four participants (technical minimum only)")
        test = set(participants[-max(1, math.ceil(len(participants) * .25)):])
        return np.where(table.participant_id.isin(test), "test", "development")
    if len(participants) != 1:
        raise ValueError("Personal calibration fits one participant at a time")
    days = sorted(table.day.unique())
    if len(days) < 3:
        raise ValueError("Personal calibration requires at least three ordered days")
    return np.where(table.day == days[-1], "test", "development")


def fit_clusters(table, output, mode="unseen"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(feature_version=FEATURE_VERSION, features=FEATURES, window_seconds=WINDOW_SECONDS,
        candidate_k=list(CANDIDATE_K), mode=mode, deployment_ready=False, status="insufficient_evidence",
        synthetic=bool(len(table) and table.synthetic.all()), cluster_meaning="descriptive input patterns; not fatigue/rest/productivity",
        limitations=["Synthetic structure is invented", "Task/device/participant confounding remains",
                     "Mouse coverage does not establish keyboard completeness", "No intervention benefit tested"])
    try:
        table = table.copy(); table["split"] = assign_split(table, mode)
        dev = table.loc[table.split == "development"]
        test = table.loc[table.split == "test"]
        X = dev[FEATURES].to_numpy(dtype=float)
        if len(dev) < 12 or len(test) < 2 or not np.isfinite(X).all() or len(np.unique(X, axis=0)) < 2:
            raise ValueError("Insufficient development/test windows or distinct activity structure")
        scaler = StandardScaler().fit(X)
        transformed = scaler.transform(X)
        candidates = []
        for k in CANDIDATE_K:
            if k >= len(X) or k > len(np.unique(X, axis=0)):
                continue
            model = KMeans(n_clusters=k, n_init=20, random_state=17).fit(transformed)
            labels = model.labels_
            if len(np.unique(labels)) < 2:
                continue
            candidates.append((float(silhouette_score(transformed, labels)), k, model))
        if not candidates:
            raise ValueError("No eligible candidate cluster structure")
        score, k, model = sorted(candidates, key=lambda value: (-value[0], value[1]))[0]
        # Stable IDs only within this model version, based on deterministic centroid order.
        order = sorted(range(k), key=lambda i: tuple(model.cluster_centers_[i]))
        mapping = {old: new for new, old in enumerate(order)}
        assignments = np.array([mapping[int(value)] for value in model.predict(scaler.transform(table[FEATURES].to_numpy(dtype=float)))])
        table["cluster_id"] = assignments
        model_version = hashlib.sha256(json.dumps(dict(feature_version=FEATURE_VERSION,
            dev_ids=sorted(dev.window_id), means=scaler.mean_.tolist(), centers=model.cluster_centers_[order].tolist()), sort_keys=True).encode()).hexdigest()[:16]
        stability = []
        rng = np.random.default_rng(29)
        for _ in range(5):
            chosen = rng.integers(0, len(X), len(X))
            boot_scaler = StandardScaler().fit(X[chosen])
            boot_model = KMeans(n_clusters=k, n_init=20, random_state=31).fit(boot_scaler.transform(X[chosen]))
            stability.append(float(adjusted_rand_score(model.predict(transformed), boot_model.predict(boot_scaler.transform(X)))))
        profiles = []
        labelled_dev = table.loc[table.split == "development"]
        for cluster_id, group in labelled_dev.groupby("cluster_id"):
            means = group[FEATURES].mean()
            name = "low observed input" if int(cluster_id) == 0 else "frequent typing" if means.keys_per_minute >= labelled_dev.keys_per_minute.median() else "more pointer activity"
            profiles.append(dict(cluster_id=int(cluster_id), description=name, windows=len(group), means=means.to_dict(),
                evaluation_context_counts=group.context.value_counts().to_dict(), participant_counts=group.participant_id.value_counts().to_dict()))
        report.update(status="technical_fit_only", model_version=model_version, selected_k=k,
            development_silhouette=score, development_candidates=[dict(k=count, silhouette=value) for value, count, _ in candidates],
            development_windows=len(dev), test_windows=len(test), bootstrap_development_ari=stability,
            constant_features=[FEATURES[i] for i, variance in enumerate(scaler.var_) if variance == 0],
            profiles=profiles, split_rule="last sorted participants held out" if mode == "unseen" else "last day held out; calibration only earlier days")
        joblib.dump(dict(feature_version=FEATURE_VERSION, model_version=model_version, scaler=scaler, kmeans=model, stable_mapping=mapping,
                         development_ids=sorted(dev.window_id)), output / "clustering.joblib")
        (output / "model.json").write_text(json.dumps(dict(feature_version=FEATURE_VERSION, features=FEATURES, window_seconds=WINDOW_SECONDS,
            model_version=model_version, synthetic=report["synthetic"], deployment_ready=False, mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(), centers=model.cluster_centers_[order].tolist()), indent=2) + "\n")
        table.to_csv(output / "assignments.csv", index=False)
        pd.DataFrame([dict(cluster_id=item["cluster_id"], description=item["description"], **item["means"]) for item in profiles]).to_csv(output / "cluster_profiles.csv", index=False)
    except ValueError as error:
        report["reason"] = str(error)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def generate_fixtures(root, people=6, days=4, windows=8):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(41); mapping = []
    for person in range(people):
        for day in range(days):
            identifier = str(uuid5(NAMESPACE_URL, f"focusfix-synthetic-v11/{person}/{day}"))
            folder = root / identifier; folder.mkdir()
            segment = str(uuid5(NAMESPACE_URL, identifier + "/segment"))
            started = datetime(2026, 10, 1 + day, 12, tzinfo=timezone.utc)
            duration = windows * WINDOW_SECONDS
            labels, events, mouse = [], [], []
            for window in range(windows):
                lower, upper = window * WINDOW_SECONDS, (window + 1) * WINDOW_SECONDS
                context = ["unknown", "work", "non_work"][window % 3]
                labelid = str(uuid5(NAMESPACE_URL, f"{identifier}/{window}"))
                labels.append(dict(id=labelid, logicalID=labelid, start=lower, end=upper, context=context,
                    source="user_action" if context != "unknown" else "automatic_default",
                    effectiveAtUTC=started.timestamp() + lower, recordedAtUTC=started.timestamp() + lower))
                active = (window + day) % 3
                rate = [0, 30, 100][active]
                for second in range(lower, upper):
                    mouse.append([second, second + 1, float(active * 2), active * 2, 0, "work", second])
                for index in range(int(rate * 5 + rng.integers(0, 5))):
                    time = lower + float(rng.uniform(0, WINDOW_SECONDS))
                    events.append([time, "delete_down" if index % 11 == 0 else "key_down", 0, "", "", "", "", "work", time])
            metadata = dict(schema_version=11, recording_format=FORMAT, required_capabilities=CAPABILITIES,
                session_id=identifier, recording_id=identifier, segment_id=segment, recording_mode="automatic",
                duration_seconds=duration, started_at_utc=started.isoformat(), status="fixture", synthetic_fixture=True)
            contract = dict(format=FORMAT, recordingID=identifier, segmentID=segment, mode="automatic", scoringTimeZone="UTC",
                elapsedOrigin=0, observedThrough=duration, finalized=True, intervals=labels, revisions=[])
            (folder / "session.json").write_text(json.dumps(metadata)); (folder / "context_intervals.json").write_text(json.dumps(contract))
            for name, header, data in [
                ("events.csv", ["elapsed_seconds", "event_type", "is_repeat", "key_hold_seconds", "scroll_delta_x", "scroll_delta_y", "scroll_precise", "activity_phase", "seconds_since_last_break"], sorted(events)),
                ("mouse_movement.csv", ["start_seconds", "end_seconds", "distance_screen_points", "movement_events", "seconds_since_last_movement", "activity_phase", "seconds_since_last_break"], mouse),
                ("breaks.csv", ["elapsed_seconds", "action", "break_id", "start_seconds", "end_seconds"], [])]:
                with (folder / name).open("w", newline="") as stream:
                    writer = csv.writer(stream); writer.writerow(header); writer.writerows(data)
            mapping.append(dict(session_id=identifier, participant_id=f"synthetic-person-{person:02}"))
    manifest = root.parent / "participants.csv"; pd.DataFrame(mapping).to_csv(manifest, index=False)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo"); demo.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit"); fit.add_argument("source", type=Path); fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True); fit.add_argument("--mode", choices=["unseen", "personal"], default="unseen")
    fit.add_argument("--allow-non-synthetic", action="store_true", help="Separate explicit approved-analysis action; never the default")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "demo":
            source = args.output / "recordings"; manifest = generate_fixtures(source)
            mode, allow = "unseen", False
        else:
            source, manifest, mode, allow = args.source, args.manifest, args.mode, args.allow_non_synthetic
        table, audit = prepare_windows(source, manifest, allow)
        table.to_csv(args.output / "windows.csv", index=False)
        (args.output / "ingestion.json").write_text(json.dumps(audit, indent=2) + "\n")
        if mode == "personal" and not table.empty:
            reports = {str(person): fit_clusters(group, args.output / f"personal-{index}", "personal") for index, (person, group) in enumerate(table.groupby("participant_id"))}
        else:
            reports = fit_clusters(table, args.output / "population", "unseen")
        print(json.dumps(reports, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
