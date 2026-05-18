from main import transform, ColumnConfig, TransformContext
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path


def test_transform():
    transform_context = TransformContext(ColumnConfig(), "85028014")

    fixture_dir = Path(__file__).parent / "golden"
    input_file_path = fixture_dir / "players-raw.csv"
    input = pl.read_csv(input_file_path, dtypes={"UID": pl.Utf8})

    output_transform, output_similar, output_shortlist = transform(
        input, transform_context
    )

    expcted_transform_file_path = fixture_dir / "replacing-pogba-1.1.csv"
    expected_transform = pl.read_csv(
        expcted_transform_file_path, dtypes={"UID": pl.Utf8}
    )
    assert_frame_equal(output_transform, expected_transform, check_exact=False)

    expcted_similar_file_path = fixture_dir / "replacing-pogba-1.3.csv"
    expected_similar = pl.read_csv(expcted_similar_file_path, dtypes={"UID": pl.Utf8})
    assert_frame_equal(output_similar, expected_similar, check_exact=False)

    expcted_shortlist_file_path = fixture_dir / "replacing-pogba-1.5.csv"
    expected_shortlist = pl.read_csv(
        expcted_shortlist_file_path, dtypes={"UID": pl.Utf8}
    )
    assert_frame_equal(
        output_shortlist.select(
            [
                "UID",
                "Name",
                "Club",
                "Similarity",
                "Chance Creation Rate",
                "Pass Completion Rate",
            ]
        ),
        expected_shortlist,
        check_exact=False,
    )
