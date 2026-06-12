from main import transform, ColumnConfig, TransformContext
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path


def test_transform():
    transform_context = TransformContext(ColumnConfig(), "85028014")

    fixture_dir = Path(__file__).parent / "golden"
    input_file_path = fixture_dir / "players_raw.csv"
    input = pl.read_csv(input_file_path, dtypes={"UID": pl.Utf8})

    output = transform(input, transform_context)

    expcted_transform_file_path = fixture_dir / "players_transformed.csv"
    expected_transform = pl.read_csv(
        expcted_transform_file_path, dtypes={"UID": pl.Utf8}
    )
    assert_frame_equal(output.transformed, expected_transform, check_exact=False)
