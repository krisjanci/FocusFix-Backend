import json
import numpy as np
import pandas as pd
import pytest

from activity_clustering import (FEATURES, assign_split, fit_clusters, generate_fixtures,
                                 load_recording, prepare_windows, rate_features)


@pytest.fixture
def generated(tmp_path):
    source = tmp_path / "recordings"
    manifest = generate_fixtures(source, people=4, days=3, windows=6)
    table, audit = prepare_windows(source, manifest)
    return source, manifest, table, audit


def test_synthetic_end_to_end_and_development_only(generated, tmp_path):
    _, _, table, audit = generated
    assert len(table) == 72 and all(item["reason"] == "included" for item in audit)
    assert set(table.context) == {"work", "non_work", "unknown"}
    first = fit_clusters(table, tmp_path / "first")
    second = fit_clusters(table, tmp_path / "second")
    assert first["status"] == "technical_fit_only"
    assert first["model_version"] == second["model_version"]
    assert first["deployment_ready"] is False and first["synthetic"] is True
    model = json.loads((tmp_path / "first/model.json").read_text())
    dev = table.loc[assign_split(table, "unseen") == "development"]
    assert np.allclose(model["mean"], dev[FEATURES].mean())
    mutated = table.copy()
    held = assign_split(mutated, "unseen") == "test"
    mutated.loc[held, FEATURES] = 1e6
    mutated.loc[:, "context"] = "non_work"  # Labels never enter fitting/selection.
    third = fit_clusters(mutated, tmp_path / "third")
    assert third["model_version"] == first["model_version"]
    assignments = pd.read_csv(tmp_path / "first/assignments.csv")
    dev_people = set(assignments.loc[assignments.split == "development", "participant_id"])
    test_people = set(assignments.loc[assignments.split == "test", "participant_id"])
    assert dev_people.isdisjoint(test_people)
    assert not set(FEATURES) & {"participant_id", "context", "work_fraction", "fatigue_0_10", "elapsed_minutes"}


def test_personal_earlier_days_only(generated, tmp_path):
    _, _, table, _ = generated
    personal = table.loc[table.participant_id == table.participant_id.iloc[0]].copy()
    report = fit_clusters(personal, tmp_path / "personal", "personal")
    assert report["status"] == "technical_fit_only"
    split = assign_split(personal, "personal")
    assert max(personal.loc[split == "development", "day"]) < min(personal.loc[split == "test", "day"])
    with pytest.raises(ValueError, match="one participant"):
        assign_split(table, "personal")


def test_stationary_bins_missing_and_checkin_exclusions(generated):
    source, manifest, _, _ = generated
    folder = next(source.iterdir())
    path = folder / "mouse_movement.csv"
    mouse = pd.read_csv(path)
    mouse.loc[mouse.start_seconds < 300, ["distance_screen_points", "movement_events"]] = 0
    mouse.to_csv(path, index=False)
    stationary, _ = prepare_windows(source, manifest)
    row = stationary.loc[(stationary.session_id == folder.name) & (stationary.start_seconds == 0)].iloc[0]
    assert row.mouse_points_per_minute == 0 and row.coverage_fraction == 1
    mouse = mouse.loc[mouse.start_seconds != 100]
    mouse.loc[(mouse.start_seconds >= 300) & (mouse.start_seconds < 301), "activity_phase"] = "check_in"
    mouse.to_csv(path, index=False)
    table, audit = prepare_windows(source, manifest)
    assert not ((table.session_id == folder.name) & (table.start_seconds.isin([0, 300]))).any()
    reasons = {item["reason"] for item in audit if item["session_id"] == folder.name}
    assert "insufficient_observed_coverage" in reasons and "check_in_or_awaiting_return" in reasons
    mouse.iloc[:0].to_csv(path, index=False)
    table, audit = prepare_windows(source, manifest)
    assert folder.name not in set(table.session_id)


def test_mixed_labels_break_and_invalid_schema(generated):
    source, manifest, _, _ = generated
    folder = next(source.iterdir())
    path = folder / "context_intervals.json"
    contract = json.loads(path.read_text())
    first = contract["intervals"][0]
    original_end = first["end"]
    first["end"] = 100
    extra = dict(first, id="00000000-0000-0000-0000-000000000099", start=100, end=original_end, context="non_work")
    contract["intervals"].insert(1, extra)
    path.write_text(json.dumps(contract))
    table, _ = prepare_windows(source, manifest)
    assert table.loc[(table.session_id == folder.name) & (table.start_seconds == 0), "context"].iloc[0] == "mixed"
    pd.DataFrame([dict(elapsed_seconds=0, action="break_start", break_id="fixture", start_seconds=0, end_seconds=""),
                  dict(elapsed_seconds=60, action="break_end", break_id="fixture", start_seconds=0, end_seconds=60)]).to_csv(folder / "breaks.csv", index=False)
    table, audit = prepare_windows(source, manifest)
    assert any(item["reason"] == "explicit_break" for item in audit)
    extra["start"] = 99
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Overlapping"):
        prepare_windows(source, manifest)
    metadata = json.loads((folder / "session.json").read_text()); metadata["schema_version"] = 10
    (folder / "session.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported"):
        load_recording(folder, "fixture")


def test_constant_features_insufficient_data_and_default_synthetic_guard(generated, tmp_path):
    source, _, table, _ = generated
    constant = table.copy(); constant[FEATURES] = 0
    report = fit_clusters(constant, tmp_path / "constant")
    assert report["status"] == "insufficient_evidence" and not (tmp_path / "constant/clustering.joblib").exists()
    report = fit_clusters(table.iloc[:2], tmp_path / "few")
    assert report["status"] == "insufficient_evidence"
    folder = next(source.iterdir())
    metadata = json.loads((folder / "session.json").read_text()); metadata["synthetic_fixture"] = False
    (folder / "session.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Generated fixtures"):
        load_recording(folder, "fixture")


def test_half_open_event_cutoff_and_float32_rates():
    events = pd.DataFrame(dict(elapsed_seconds=[0, 299.9, 300], event_type=["key_down"] * 3, activity_phase=["work"] * 3))
    mouse = pd.DataFrame(dict(start_seconds=range(300), end_seconds=range(1, 301), activity_phase=["work"] * 300,
                              distance_screen_points=[0] * 300, movement_events=[0] * 300))
    features = rate_features(events, mouse, 0, 300)
    assert features["keys_per_minute"] == float(np.float32(.4))
