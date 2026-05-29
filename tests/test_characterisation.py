import filecmp
from pathlib import Path
import pytest
import main


def test_characterisation(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    
    (tmp_path / "data").mkdir()

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