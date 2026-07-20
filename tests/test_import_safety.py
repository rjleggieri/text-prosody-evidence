import subprocess
import sys


def test_import_has_no_data_or_download_side_effects() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import text_prosody.external_benchmark, text_prosody.external_evaluation, "
            "text_prosody.evidence_bundle",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
