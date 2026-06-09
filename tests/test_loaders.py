from loaders import write_targets
from config import LoadTarget
from pathlib import Path
import polars as pl
import filecmp


def test_write_targets(tmp_path, monkeypatch):
    # given targets
    monkeypatch.chdir(tmp_path)

    tmp_dir = tmp_path / "data"

    tmp_dir.mkdir()

    expected_file_path = tmp_dir / "players_raw.csv"

    fixture_dir = Path(__file__).parent / "golden"
    input_file_path = fixture_dir / "players_raw.csv"
    input_df = pl.read_csv(input_file_path, dtypes={"UID": pl.Utf8})
    input_targets = [LoadTarget(None, input_df, expected_file_path)]

    # run funciton
    write_targets(input_targets)
    # check output
    assert expected_file_path.exists()

    assert pl.read_csv(input_file_path).equals(pl.read_csv(expected_file_path))
