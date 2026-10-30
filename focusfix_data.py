"""Versioned FocusFix recording ingestion and causal work-window features."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import pandas as pd

FEATURE_VERSION = "focusfix_work_window_v1"
FEATURES = [
    "elapsed_minutes", "work_fraction", "keys_per_minute", "deletes_per_minute",
    "clicks_per_minute", "switches_per_minute", "scrolls_per_minute",
    "mouse_points_per_minute", "mouse_events_per_minute", "mean_key_hold_seconds",
]
TARGETS = ["fatigue_0_10", "productivity_0_10"]
FILES = {"session.json", "events.csv", "mouse_movement.csv", "surveys.csv",
         "breaks.csv", "survey_prompts.csv"}


@contextmanager
def recording_root(source):
    """Accept a recording tree or research ZIP; never extract arbitrary paths."""
    source = Path(source)
    if source.is_dir():
        yield source
        return
    with TemporaryDirectory(prefix="focusfix-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if sum(e.file_size for e in entries) > 512 * 1024 * 1024:
                raise ValueError("ZIP exceeds the 512 MiB uncompressed limit")
            seen = set()
            for entry in entries:
                path = Path(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                    raise ValueError("Unsafe ZIP path")
                if entry.is_dir() or path.name not in FILES:
                    continue
                if entry.filename in seen or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Duplicate or symlink ZIP entry")
                seen.add(entry.filename)
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(entry))
        yield root


def read_table(folder, name, columns, numeric):
    table = pd.read_csv(folder / name, keep_default_na=False)
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    for column in numeric:
        original = table[column].mask(table[column] == "", np.nan)
        table[column] = pd.to_numeric(original, errors="raise")
        if np.isinf(table[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{name}: infinite {column}")
    return table


def window_features(events, mouse, cutoff, window_seconds=300):
    """Use [cutoff-window, cutoff), complete work mouse bins only.

    Work exposure is measured by recorded mouse bins, including stationary bins.
    No interpolation through pauses, check-ins, missing bins or future data.
    The app must apply this exact contract when computing live model inputs.
    """
    if not np.isfinite(cutoff) or cutoff <= 0 or not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("Cutoff and window must be finite and positive")
    start = max(0.0, cutoff - window_seconds)
    bins = mouse.loc[(mouse.start_seconds >= start) & (mouse.end_seconds <= cutoff)
                     & (mouse.activity_phase == "work")].sort_values("start_seconds")
    exposure = float((bins.end_seconds - bins.start_seconds).sum())
    if exposure < 30:
        return None
    # Events must fall within observed work bins, so gaps don't inflate rates.
    work = events.loc[(events.elapsed_seconds >= start) & (events.elapsed_seconds < cutoff)
                      & (events.activity_phase == "work")]
    starts, ends = bins.start_seconds.to_numpy(), bins.end_seconds.to_numpy()
    positions = np.searchsorted(starts, work.elapsed_seconds, side="right") - 1
    observed = (positions >= 0) & (work.elapsed_seconds.to_numpy() < ends[np.maximum(positions, 0)])
    work = work.loc[observed]
    rate = 60 / exposure
    types = work.event_type
    holds = work.loc[types.isin(["key_up", "delete_up"]), "key_hold_seconds"].dropna()
    return dict(zip(FEATURES, [
        cutoff / 60, exposure / (cutoff - start),
        float(types.isin(["key_down", "delete_down"]).sum()) * rate,
        float((types == "delete_down").sum()) * rate,
        float(types.isin(["left_click_down", "right_click_down", "other_click_down"]).sum()) * rate,
        float((types == "app_switch").sum()) * rate,
        float((types == "scroll").sum()) * rate,
        float(bins.distance_screen_points.sum()) * rate,
        float(bins.movement_events.sum()) * rate,
        float(holds.mean()) if len(holds) else 0.0,
    ]))


def load_session(folder, participant_id, window_seconds=300):
    metadata = json.loads((folder / "session.json").read_text())
    if metadata.get("schema_version") not in (5, 6, 7, 8):
        raise ValueError("Only recording schemas 5–8 are supported; older surveys have different meanings")
    if metadata.get("check_in_version") != "focusfix_check_in_v2":
        raise ValueError("Only focusfix_check_in_v2 ratings are supported")
    duration = metadata.get("duration_seconds")
    if metadata.get("status") == "recording" or not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Recording must be finished with a positive duration")
    session_id = str(metadata["session_id"])
    events = read_table(folder, "events.csv",
                        ["elapsed_seconds", "event_type", "activity_phase", "key_hold_seconds"],
                        ["elapsed_seconds", "key_hold_seconds"])
    mouse = read_table(folder, "mouse_movement.csv",
                       ["start_seconds", "end_seconds", "activity_phase", "distance_screen_points", "movement_events"],
                       ["start_seconds", "end_seconds", "distance_screen_points", "movement_events"])
    surveys = read_table(folder, "surveys.csv",
                         ["session_id", "check_in_id", "question_version", "context", "status",
                          "shown_elapsed_seconds", "answered_elapsed_seconds"] + TARGETS,
                         ["shown_elapsed_seconds", "answered_elapsed_seconds"] + TARGETS)
    for table, columns in [(events, ["elapsed_seconds"]), (mouse, ["start_seconds", "end_seconds", "distance_screen_points", "movement_events"])]:
        if table[columns].isna().any().any() or (table[columns] < 0).any().any():
            raise ValueError("Missing or negative activity measurements")
    if (events.elapsed_seconds > duration).any() or (mouse.end_seconds > duration).any():
        raise ValueError("Activity extends beyond session duration")
    if (events.key_hold_seconds.dropna() < 0).any():
        raise ValueError("Negative key hold duration")
    ordered = mouse.sort_values("start_seconds")
    if (ordered.end_seconds <= ordered.start_seconds).any() or (ordered.start_seconds.to_numpy()[1:] < ordered.end_seconds.to_numpy()[:-1]).any():
        raise ValueError("Overlapping or invalid mouse bins")
    if not events.activity_phase.isin(["work", "check_in", "awaiting_return"]).all() or not mouse.activity_phase.isin(["work", "check_in", "awaiting_return"]).all():
        raise ValueError("Unknown activity phase")
    if surveys.check_in_id.duplicated().any() or (surveys.check_in_id == "").any():
        raise ValueError("Missing or duplicate check-in ID")
    if not (surveys.session_id == session_id).all() or not (surveys.question_version == "focusfix_check_in_v2").all():
        raise ValueError("Survey session/version mismatch")
    if not surveys.status.isin(["answered", "skipped", "cancelled"]).all() or not surveys.context.isin(["session_start", "during_work", "before_break"]).all():
        raise ValueError("Unknown survey status/context")
    for target in TARGETS:
        ratings = surveys[target].dropna()
        if ((ratings < 0) | (ratings > 10) | (ratings % 1 != 0)).any():
            raise ValueError(f"Invalid rating: {target}")
    if surveys[["shown_elapsed_seconds", "answered_elapsed_seconds"]].isna().any().any() or (surveys.shown_elapsed_seconds > surveys.answered_elapsed_seconds).any() or (surveys.answered_elapsed_seconds > duration).any():
        raise ValueError("Invalid survey timing")
    if surveys.loc[surveys.status != "answered", TARGETS].notna().any().any():
        raise ValueError("Unanswered surveys must have blank ratings")
    rows, skipped = [], 0
    for survey in surveys.itertuples(index=False):
        if survey.status != "answered" or survey.context == "session_start":
            skipped += 1
            continue
        if pd.isna(survey.fatigue_0_10) or pd.isna(survey.productivity_0_10):
            raise ValueError("Answered work check-in must contain both ratings")
        features = window_features(events, mouse, survey.shown_elapsed_seconds, window_seconds)
        if features is None:
            skipped += 1
            continue
        rows.append({"participant_id": participant_id, "session_id": session_id,
                     "check_in_id": survey.check_in_id, **features,
                     **{t: getattr(survey, t) for t in TARGETS}})
    return session_id, rows, skipped


def prepare(source, participant_id=None, manifest=None, window_seconds=300, skip_invalid=False):
    """Manifest CSV maps session_id to pseudonymous participant_id across all sessions."""
    if not np.isfinite(window_seconds) or window_seconds < 30:
        raise ValueError("Window must be finite and at least 30 seconds")
    if bool(participant_id) == bool(manifest):
        raise ValueError("Supply exactly one of --participant-id or --manifest")
    identities = None
    if manifest:
        mapping = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        if not {"session_id", "participant_id"} <= set(mapping.columns) or mapping.session_id.duplicated().any() or (mapping[["session_id", "participant_id"]] == "").any().any():
            raise ValueError("Manifest needs unique nonempty session_id and participant_id")
        identities = mapping.set_index("session_id").participant_id.to_dict()
    rows, report, seen = [], [], set()
    with recording_root(source) as root:
        sessions = sorted(root.rglob("session.json"))
        if not sessions:
            raise ValueError("No session.json recordings found")
        for path in sessions:
            identity = str(json.loads(path.read_text())["session_id"])
            if identity in seen:
                raise ValueError("Duplicate session across input folders")
            seen.add(identity)
            if identities is not None and identity not in identities:
                raise ValueError(f"Manifest has no participant for session {identity}")
            try:
                identity, examples, skipped = load_session(path.parent, identities[identity] if identities is not None else participant_id, window_seconds)
            except (ValueError, KeyError, OSError) as error:
                if not skip_invalid:
                    raise ValueError(f"Session {identity}: {error}") from error
                report.append({"session_id": identity, "training_rows": 0, "excluded_session": str(error)})
                continue
            rows.extend(examples)
            report.append({"session_id": identity, "training_rows": len(examples), "excluded_check_ins": skipped, "synthetic": bool(json.loads(path.read_text()).get("synthetic", False))})
    return pd.DataFrame(rows, columns=["participant_id", "session_id", "check_in_id"] + FEATURES + TARGETS), report
