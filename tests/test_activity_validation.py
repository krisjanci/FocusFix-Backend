from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from activity_clustering import FEATURE_VERSION
from activity_validation import earlier_personal_ratings, evaluate, join_ratings, synthetic_ratings


def assignments(people=12, days=4, windows=10, personal=False):
    rows = []
    for person in range(people):
        for day in range(days):
            for index in range(windows):
                start = index * 300
                date = datetime(2026, 10, day + 1, 12, tzinfo=timezone.utc) + timedelta(seconds=start)
                rows.append(dict(window_id=f"p{person}-d{day}-w{index}", session_id=f"p{person}-d{day}", participant_id=f"p{person:02}",
                    start_seconds=start, end_seconds=start + 300, start_utc=date.isoformat(), day=date.date().isoformat(),
                    context=["work", "non_work", "unknown"][index % 3], cluster_id=index % 3,
                    work_fraction=1 if index % 3 == 0 else 0, labels_recorded_at_utc=date.timestamp(),
                    split="test" if (day == days - 1 if personal else person >= people - 3) else "development"))
    return pd.DataFrame(rows)


def report(mode="unseen"):
    return dict(feature_version=FEATURE_VERSION, status="technical_fit_only", mode=mode, synthetic=True,
                model_version="synthetic-frozen", development_silhouette=.5, bootstrap_development_ari=[.8, .9])


def test_independent_join_precedes_prompt_and_missing_stays_missing():
    windows = assignments(people=1, days=1, windows=3)
    ratings = synthetic_ratings(windows)
    table, audit = join_ratings(windows, ratings)
    assert len(table) == 2 and any(item["reason"] == "missing_or_skipped" for item in audit)
    assert (table.window_end_seconds <= table.recall_end).all()
    assert (table.recall_end <= table.request_elapsed).all()
    # Invalid recall cannot consume future windows, prompt interaction, or a label as an answer.
    ratings.loc[1, "recall_end"] = ratings.loc[1, "request_elapsed"] + 1
    ratings.loc[2, "independent"] = False
    table, audit = join_ratings(windows, ratings)
    assert table.empty
    assert {item["reason"] for item in audit} >= {"invalid_recall_or_rating", "not_independent"}


def test_comparison_and_participant_uncertainty(tmp_path):
    windows = assignments()
    ratings = synthetic_ratings(windows)
    result = evaluate(windows, ratings, report(), tmp_path / "evaluation")
    assert result["status"] == "synthetic_technical_comparison"
    assert set(result["held_out_mae"]) == {"training_mean", "context_time_prior_work", "plus_cluster"}
    assert len(result["participant_bootstrap_mae_improvement_interval"]) == 2
    assert result["intervention_benefit"] == "not_tested"
    assert result["cluster_structure"]["real_world_stability"] == "not_established"
    assert result["missingness"]
    # Final-test changes cannot change the training mean benchmark.
    changed = ratings.copy(); changed.loc[changed.participant_id.isin(["p09", "p10", "p11"]), "fatigue"] = 10
    changed.loc[changed.status != "answered", "fatigue"] = np.nan
    evaluate(windows, changed, report(), tmp_path / "changed")
    before = pd.read_csv(tmp_path / "evaluation/held_out_predictions.csv")
    after = pd.read_csv(tmp_path / "changed/held_out_predictions.csv")
    assert np.array_equal(before.training_mean, after.training_mean)


def test_insufficient_evidence_has_no_accuracy_claim(tmp_path):
    windows = assignments(people=4, days=1, windows=2)
    result = evaluate(windows, synthetic_ratings(windows), report(), tmp_path / "few")
    assert result["status"] == "insufficient_evidence" and "held_out_mae" not in result
    assert not (tmp_path / "few/held_out_predictions.csv").exists()


def test_personal_baseline_uses_only_strictly_earlier_ratings(tmp_path):
    windows = assignments(people=1, days=4, windows=30, personal=True)
    ratings = synthetic_ratings(windows)
    table, _ = join_ratings(windows, ratings)
    features = earlier_personal_ratings(table)
    future = table.copy(); last = future.index[-1]; future.loc[last, "fatigue"] = 10
    mutated = earlier_personal_ratings(future)
    assert np.array_equal(features.prior_fatigue.iloc[:-1], mutated.prior_fatigue.iloc[:-1])
    first = features.sort_values("request_utc").iloc[0]
    assert first.prior_rating_available == 0
    result = evaluate(windows, ratings, report("personal"), tmp_path / "later", "later-time")
    assert result["status"] == "synthetic_technical_comparison" and "uncertainty" in result
    with pytest.raises(ValueError, match="split"):
        evaluate(windows, ratings, report(), tmp_path / "wrong", "later-time")


def test_later_context_corrections_do_not_enter_prior_work():
    windows = assignments(people=1, days=1, windows=3)
    ratings = synthetic_ratings(windows)
    windows["labels_recorded_at_utc"] = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()
    table, _ = join_ratings(windows, ratings)
    assert (table.prior_known_work_minutes == 0).all() and (table.prior_labels_unavailable > 0).all()


def test_study_adapter_preserves_zero_and_missing_with_exact_schedule(tmp_path):
    import json
    from activity_validation import read_study_ratings
    folder = tmp_path / "recordings/fixture"; folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps(dict(schema_version=11, synthetic_fixture=True)))
    pd.DataFrame([dict(session_id="fixture", participant_id="synthetic-person")]).to_csv(tmp_path / "manifest.csv", index=False)
    entries = []
    answers = []
    for index, status in enumerate(["answered", "skipped"]):
        prompt = dict(id=f"check-{index}", scheduleVersion="synthetic.v1", configurationScope="synthetic", contextAtPrompt="non_work",
                      requestedElapsed=600 + index * 600, recallStart=300 + index * 600, recallEnd=600 + index * 600,
                      requestedAtUTC=1793620800 + index * 600)
        entries.append(dict(prompt=prompt, action="requested", elapsed=prompt["requestedElapsed"]))
        answers.append(dict(check_in_id=prompt["id"], status=status, fatigue_0_10=0 if index == 0 else "",
                            answered_elapsed_seconds=prompt["requestedElapsed"] + 20))
    (folder / "study_check_ins.jsonl").write_text("\n".join(json.dumps(item) for item in entries))
    pd.DataFrame(answers).to_csv(folder / "surveys.csv", index=False)
    loaded = read_study_ratings(tmp_path / "recordings", tmp_path / "manifest.csv", "synthetic.v1")
    assert loaded.fatigue.iloc[0] == 0 and np.isnan(loaded.fatigue.iloc[1])
    assert read_study_ratings(tmp_path / "recordings", tmp_path / "manifest.csv", "other.v1").empty
    (folder / "session.json").write_text(json.dumps(dict(schema_version=11, synthetic_fixture=False)))
    with pytest.raises(ValueError, match="reviewed study scope"):
        read_study_ratings(tmp_path / "recordings", tmp_path / "manifest.csv", "synthetic.v1", True)



def test_late_answers_are_unavailable_to_earlier_personal_predictions():
    windows = assignments(people=1, days=1, windows=4)
    ratings = synthetic_ratings(windows)
    table, _ = join_ratings(windows, ratings)
    table = table.sort_values("request_utc").reset_index(drop=True)
    table.loc[0, "answered_utc"] = "2027-01-01T00:00:00+00:00"
    features = earlier_personal_ratings(table)
    assert features.loc[1, "prior_rating_available"] == 0
    assert features.loc[2, "prior_fatigue"] == table.loc[1, "fatigue"]
