"""Grouped model selection, untouched test groups and reproducible artifacts."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from coreml_export import export_model
from focusfix_data import FEATURES, FEATURE_VERSION, TARGETS


def candidates():
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "decision_tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=80, max_depth=2, min_samples_leaf=3, learning_rate=0.05, random_state=42),
    }


def clipped_predict(model, x):
    return np.clip(model.predict(x), 0, 10)


def train_models(table, output, group_by="participant", models=None, export=True,
                 window_seconds=300, synthetic=False):
    group_column = {"participant": "participant_id", "session": "session_id"}[group_by]
    required = ["participant_id", "session_id", "check_in_id"] + FEATURES + TARGETS
    if not set(required) <= set(table.columns):
        raise ValueError("Training table is missing required columns")
    table = table.reset_index(drop=True)
    if table[["participant_id", "session_id", "check_in_id"]].isna().any().any() or (table[["participant_id", "session_id", "check_in_id"]] == "").any().any():
        raise ValueError("Training identifiers must be nonempty")
    if table.duplicated(["session_id", "check_in_id"]).any() or (table.groupby("session_id").participant_id.nunique() > 1).any():
        raise ValueError("Duplicate training check-in or inconsistent session identity")
    numbers = table[FEATURES + TARGETS].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any() or (table[TARGETS] > 10).any().any():
        raise ValueError("Training values must be finite, nonnegative; ratings must be 0–10")
    groups = table[group_column]
    if groups.nunique() < 4:
        raise ValueError(f"Need at least four distinct {group_column} groups: one untouched test group and at least three development groups. Collect more labelled recordings; do not duplicate rows.")
    templates = candidates()
    if models:
        unknown = set(models) - templates.keys()
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        templates = {key: model for key, model in templates.items() if key == "mean_baseline" or key in models}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory, avoiding stale model artifacts")
    output.mkdir(parents=True, exist_ok=True)
    x = table[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features exceed float32 range")
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(x, groups=groups))
    dev_groups = groups.iloc[development]
    folds = list(GroupKFold(n_splits=min(5, dev_groups.nunique())).split(x[development], groups=dev_groups))
    assignments = table[["participant_id", "session_id", "check_in_id"]].copy()
    assignments["split"] = "development"
    assignments.loc[test, "split"] = "test"
    assignments.to_csv(output / "split_assignments.csv", index=False)
    report = {"feature_version": FEATURE_VERSION, "features": FEATURES,
              "window_seconds": window_seconds, "minimum_work_seconds": 30,
              "group_by": group_by, "synthetic": synthetic,
              "evaluation_scope": "unseen people" if group_by == "participant" else "unseen sessions; does not establish accuracy for new people",
              "rows": len(table), "groups": int(groups.nunique()),
              "development_groups": int(dev_groups.nunique()), "test_groups": int(groups.iloc[test].nunique()),
              "data_sha256": hashlib.sha256(table.to_csv(index=False).encode()).hexdigest(),
              "targets": {}, "deployment_ready": False,
              "warning": "Experimental. Model selection uses grouped development CV; only selected model and baseline see the test groups. Final artifacts refit all data after evaluation. No causal break recommendation or medical claim."}
    comparisons = []
    predictions = []
    for target in TARGETS:
        y = table[target].to_numpy(dtype=float)
        scores = {}
        for name, template in templates.items():
            held_out = np.full(len(development), np.nan)
            for fit, validation in folds:
                fitted = clone(template).fit(x[development[fit]], y[development[fit]])
                held_out[validation] = clipped_predict(fitted, x[development[validation]])
            scores[name] = float(mean_absolute_error(y[development], held_out))
            for row, prediction in zip(development, held_out):
                predictions.append({"row": int(row), "target": target, "model": name, "split": "development_cv", "actual": y[row], "prediction": prediction})
        selected = min(scores, key=scores.get)  # insertion order favors baseline on ties
        test_errors = {}
        for name in dict.fromkeys(["mean_baseline", selected]):
            fitted = clone(templates[name]).fit(x[development], y[development])
            predicted = clipped_predict(fitted, x[test])
            test_errors[name] = float(mean_absolute_error(y[test], predicted))
            for row, prediction in zip(test, predicted):
                predictions.append({"row": int(row), "target": target, "model": name, "split": "test", "actual": y[row], "prediction": prediction})
        target_report = {"selected_model": selected, "development_mae": scores[selected],
                         "baseline_development_mae": scores["mean_baseline"],
                         "test_mae": test_errors[selected], "baseline_test_mae": test_errors["mean_baseline"],
                         "beats_test_baseline": test_errors[selected] < test_errors["mean_baseline"], "artifacts": {}}
        for name, template in templates.items():
            comparisons.append({"target": target, "model": name, "development_mae": scores[name], "selected": name == selected})
            final = clone(template).fit(x, y)
            stem = f"{target}_{name}"
            joblib.dump(final, output / f"{stem}.joblib")
            if export:
                parity_rows = x if len(x) <= 128 else x[np.linspace(0, len(x)-1, 128, dtype=int)]
                verification = export_model(final, FEATURES, target, output / f"{stem}.mlmodel",
                                            {"feature_version": FEATURE_VERSION, "window_seconds": window_seconds,
                                             "synthetic": synthetic, "experimental": True}, parity_rows)
                target_report["artifacts"][name] = {"file": f"{stem}.mlmodel", **verification}
        report["targets"][target] = target_report
    pd.DataFrame(comparisons).to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "held_out_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
