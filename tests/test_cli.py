from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_recordings_through_training_and_exports(tmp_path):
    destination = tmp_path / "demo"
    result = subprocess.run([sys.executable, str(ROOT / "train.py"), "demo", "--output", str(destination)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / "models/report.json").read_text())
    assert report["synthetic"] is True
    assert len(list((destination / "models").glob("*.mlmodel"))) == 12
    for target in report["targets"].values():
        assert len(target["artifacts"]) == 6
    repeated = subprocess.run([sys.executable, str(ROOT / "train.py"), "demo", "--output", str(destination)],
                              capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 2
    assert "new or empty" in repeated.stderr
