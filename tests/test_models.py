import json
import platform

import numpy as np
import pandas as pd
import pytest

from coreml_export import export_model
from focusfix_data import FEATURES, TARGETS
from modeling import candidates, train_models


def sample_table():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, (60, len(FEATURES))).astype(np.float32)
    table = pd.DataFrame(x, columns=FEATURES)
    table["participant_id"] = np.repeat([f"p{n}" for n in range(6)], 10)
    table["session_id"] = table.participant_id + "-session"
    table["check_in_id"] = [f"c{n}" for n in range(60)]
    table[TARGETS[0]] = np.round(x[:, 0])
    table[TARGETS[1]] = np.round(10 - x[:, 0])
    return table


def test_group_separation_and_results(tmp_path):
    table = sample_table()
    report = train_models(table, tmp_path / "run", export=False)
    assignments = pd.read_csv(tmp_path / "run/split_assignments.csv")
    assert assignments.groupby("participant_id").split.nunique().max() == 1
    comparison = pd.read_csv(tmp_path / "run/model_comparison.csv")
    assert len(comparison) == 12
    assert report["deployment_ready"] is False
    for target in TARGETS:
        assert report["targets"][target]["selected_model"] != "mean_baseline"
        assert report["targets"][target]["test_mae"] < report["targets"][target]["baseline_test_mae"]
    with pytest.raises(ValueError, match="empty directory"):
        train_models(table, tmp_path / "run", export=False)


def test_not_enough_people_and_invalid_data(tmp_path):
    table = sample_table()
    table["participant_id"] = "same-person"
    with pytest.raises(ValueError, match="at least four"):
        train_models(table, tmp_path / "run", export=False)
    table = sample_table(); table.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train_models(table, tmp_path / "run", export=False)


def test_baseline_wins_ties(tmp_path):
    table = sample_table(); table[TARGETS] = 5
    report = train_models(table, tmp_path / "run", export=False)
    assert all(r["selected_model"] == "mean_baseline" for r in report["targets"].values())


@pytest.mark.parametrize("name", list(candidates()))
def test_every_model_coreml_parity(name, tmp_path):
    table = sample_table()
    x = table[FEATURES].to_numpy(dtype=np.float32)
    model = candidates()[name].fit(x, table[TARGETS[0]])
    destination = tmp_path / f"{name}.mlmodel"
    result = export_model(model, FEATURES, TARGETS[0], destination, {"synthetic": True}, x)
    assert destination.exists()
    if platform.system() == "Darwin":
        assert result["runtime_verified"]
        assert result["max_absolute_error"] < 1e-3
    import coremltools as ct
    exported = ct.models.MLModel(str(destination), skip_model_load=True)
    assert json.loads(exported.user_defined_metadata["feature_order"]) == FEATURES
    assert exported.get_spec().description.output[0].name == "raw_rating"
