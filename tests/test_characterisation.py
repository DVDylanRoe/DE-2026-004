import filecmp
from pathlib import Path
import pytest
import main
import sys


def test_characterisation(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    (tmp_path / "data").mkdir()

    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setattr(sys, "argv", ["prog"])

    main.main(load_sf=False)

    expected_files = [
        "players_raw.csv",
        "players_transformed.csv",
    ]

    golden_dir = Path(__file__).parent / "golden"

    for filename in expected_files:
        produced = tmp_path / "data" / filename
        golden = golden_dir / filename

        assert produced.exists(), f"Expected output {filename} was not created"
        assert golden.exists(), f"Golden file missing: {golden}"

        assert filecmp.cmp(produced, golden, shallow=False), (
            f"Output mismatch for {filename}. "
            "If this change is intentional, update the golden master."
        )
