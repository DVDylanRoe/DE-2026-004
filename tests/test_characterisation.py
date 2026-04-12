import filecmp
from pathlib import Path
import pytest
import replacing_pogba


def test_characterisation(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    replacing_pogba.main()

    expected_files = [
        "players-raw.csv",
        "replacing-pogba-1.1.csv",
        "replacing-pogba-1.3.csv",
        "replacing-pogba-1.5.csv",
    ]

    golden_dir = Path(__file__).parent / "golden"

    for filename in expected_files:
        produced = tmp_path / filename
        golden = golden_dir / filename

        assert produced.exists(), f"Expected output {filename} was not created"

        assert golden.exists(), f"Golden file missing: {golden}"

        assert filecmp.cmp(produced, golden, shallow=False), (
            f"Output mismatch for {filename}. "
            "If this change is intentional, update the golden master."
        )
