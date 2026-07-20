import json
import subprocess
import sys
from pathlib import Path


def test_cli_help_has_no_side_effects() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "text_prosody.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "synthetic-reference" in result.stdout


def test_cli_writes_aggregate_synthetic_artifacts_only(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "text_prosody.cli",
            "synthetic-reference",
            "--output",
            str(output),
            "--bootstrap-replicates",
            "100",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "metrics.json",
        "reference_calibration.json",
        "run_manifest.json",
    ]
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SYNTHETIC_SOFTWARE_VERIFICATION_ONLY"
    assert manifest["scientific_validation"] is False
    assert manifest["real_data_index_frozen"] is False
    assert manifest["item_level_outputs_persisted"] is False
