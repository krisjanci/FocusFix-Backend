"""Deterministic, explicitly synthetic schema-8 sessions for pipeline verification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate(destination, participants=6, sessions=2, seed=42):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Synthetic destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    for person in range(participants):
        for session in range(sessions):
            identity = f"synthetic-p{person}-s{session}"
            folder = destination / identity
            folder.mkdir()
            (folder / "session.json").write_text(json.dumps({
                "schema_version": 8, "session_id": identity,
                "check_in_version": "focusfix_check_in_v2", "duration_seconds": 1201,
                "status": "user_stop", "synthetic": True,
            }))
            events, mouse, surveys = [], [], []
            for block in range(4):
                activity = int(rng.integers(1, 10))
                start = block * 300
                for second in range(start, start + 300):
                    mouse.append([second, second + 1, activity * 2, activity, 0, "work", second + 1])
                    for key in range(activity):
                        events.append([second + key / 20, "delete_down" if key == 0 else "key_down", 0, "", "", "", "", "work", second])
                surveys.append([identity, f"check-{block}", "focusfix_check_in_v2", "during_work", "answered", "", "", "", start + 300, start + 300.5,
                                int(np.clip(10 - activity + rng.normal(0, 0.3), 0, 10)), activity])
            pd.DataFrame(events, columns="elapsed_seconds,event_type,is_repeat,key_hold_seconds,scroll_delta_x,scroll_delta_y,scroll_precise,activity_phase,seconds_since_last_break".split(",")).to_csv(folder / "events.csv", index=False)
            pd.DataFrame(mouse, columns="start_seconds,end_seconds,distance_screen_points,movement_events,seconds_since_last_movement,activity_phase,seconds_since_last_break".split(",")).to_csv(folder / "mouse_movement.csv", index=False)
            pd.DataFrame(surveys, columns="session_id,check_in_id,question_version,context,status,break_id,shown_at_utc,answered_at_utc,shown_elapsed_seconds,answered_elapsed_seconds,fatigue_0_10,productivity_0_10".split(",")).to_csv(folder / "surveys.csv", index=False)
            manifest.append({"session_id": identity, "participant_id": f"synthetic-p{person}"})
    pd.DataFrame(manifest).to_csv(destination / "participants.csv", index=False)
    return destination / "participants.csv"
