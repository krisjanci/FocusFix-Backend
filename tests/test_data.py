import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from focusfix_data import FEATURES, load_session, prepare, recording_root, window_features
from synthetic import generate


@pytest.fixture
def recording(tmp_path):
    generate(tmp_path / "recordings", participants=1, sessions=1)
    return tmp_path / "recordings" / "synthetic-p0-s0"


def test_real_format_and_causal_window(recording):
    identity, rows, skipped = load_session(recording, "person")
    assert len(rows) == 4 and skipped == 0
    original = rows[0]
    events = pd.read_csv(recording / "events.csv")
    # Events at/after the check-in cannot influence the preceding prediction.
    events.loc[events.elapsed_seconds >= 300, "event_type"] = "app_switch"
    events.to_csv(recording / "events.csv", index=False)
    assert load_session(recording, "person")[1][0] == original
    assert original["keys_per_minute"] > 0
    assert original["switches_per_minute"] == 0


def test_work_only_and_missing_coverage(recording):
    events = pd.read_csv(recording / "events.csv")
    mouse = pd.read_csv(recording / "mouse_movement.csv")
    original = window_features(events, mouse, 300)
    mouse.loc[mouse.start_seconds < 150, "activity_phase"] = "check_in"
    events.loc[events.elapsed_seconds < 150, "activity_phase"] = "check_in"
    result = window_features(events, mouse, 300)
    assert result["work_fraction"] == 0.5
    assert result["keys_per_minute"] == original["keys_per_minute"]
    assert window_features(events, mouse.iloc[:20], 300) is None
    # No events from missing mouse coverage may inflate the work rates.
    mouse = mouse.loc[mouse.start_seconds >= 150]
    events.loc[events.elapsed_seconds < 150, "activity_phase"] = "work"
    assert window_features(events, mouse, 300) == result


def test_skipped_and_start_surveys_are_not_zero_labels(recording):
    survey = pd.read_csv(recording / "surveys.csv")
    survey.loc[0, "status"] = "skipped"
    survey.loc[0, ["fatigue_0_10", "productivity_0_10"]] = np.nan
    survey.loc[1, "context"] = "session_start"
    survey.loc[1, "productivity_0_10"] = np.nan
    survey.to_csv(recording / "surveys.csv", index=False)
    assert len(load_session(recording, "person")[1]) == 2


@pytest.mark.parametrize("mutation,match", [
    ("overlap", "Overlapping"), ("negative", "negative"), ("bad_rating", "Invalid rating"),
    ("duplicate", "duplicate"), ("version", "different meanings"), ("unfinished", "finished"),
    ("nan_time", "Missing"), ("bad_phase", "Unknown activity"),
])
def test_invalid_recordings_rejected(recording, mutation, match):
    if mutation in ("version", "unfinished"):
        p = recording / "session.json"
        value = json.loads(p.read_text())
        value["schema_version" if mutation == "version" else "status"] = 4 if mutation == "version" else "recording"
        p.write_text(json.dumps(value))
    elif mutation in ("overlap", "negative", "nan_time", "bad_phase"):
        p = recording / "mouse_movement.csv"
        value = pd.read_csv(p)
        if mutation == "overlap": value.loc[1, "start_seconds"] = 0
        if mutation == "negative": value.loc[0, "distance_screen_points"] = -1
        if mutation == "nan_time": value.loc[0, "start_seconds"] = np.nan
        if mutation == "bad_phase": value.loc[0, "activity_phase"] = "unknown"
        value.to_csv(p, index=False)
    else:
        p = recording / "surveys.csv"
        value = pd.read_csv(p)
        if mutation == "bad_rating": value.loc[0, "fatigue_0_10"] = 11
        if mutation == "duplicate": value.loc[1, "check_in_id"] = value.loc[0, "check_in_id"]
        value.to_csv(p, index=False)
    with pytest.raises(ValueError, match=match):
        load_session(recording, "person")


def test_folder_and_zip_equal(recording, tmp_path):
    direct, _ = prepare(recording, participant_id="person")
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in recording.iterdir(): z.write(path, f"export/{path.name}")
    zipped, _ = prepare(archive, participant_id="person")
    pd.testing.assert_frame_equal(direct, zipped)
    assert np.isfinite(direct[FEATURES]).all().all()


def test_bad_zip_and_missing_manifest_rejected(recording, tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z: z.writestr("../session.json", "{}")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        with recording_root(archive): pass
    manifest = tmp_path / "mapping.csv"
    manifest.write_text("session_id,participant_id\nother,person\n")
    with pytest.raises(ValueError, match="no participant"):
        prepare(recording, manifest=manifest)


def test_exclusion_audit_and_window_validation(recording):
    metadata = recording / "session.json"
    value = json.loads(metadata.read_text()); value["status"] = "recording"
    metadata.write_text(json.dumps(value))
    table, audit = prepare(recording, participant_id="person", skip_invalid=True)
    assert table.empty and "finished" in audit[0]["excluded_session"]
    with pytest.raises(ValueError, match="Window"):
        prepare(recording, participant_id="person", window_seconds=float("nan"))


def test_csv_timestamp_rounding(recording):
    metadata = recording / "session.json"
    value = json.loads(metadata.read_text()); value["duration_seconds"] = 1199.9999996
    metadata.write_text(json.dumps(value))
    surveys = pd.read_csv(recording / "surveys.csv")
    surveys.loc[3, "answered_elapsed_seconds"] = 1200
    surveys.to_csv(recording / "surveys.csv", index=False)
    assert len(load_session(recording, "person")[1]) == 4
