import pytest
import polars as pl
from config import (
    ColumnConfig,
    TransformContext,
    Baseline,
)
from shortlist import (
    create_shortlist,
    find_player_baseline,
    filter_shortlist,
)


def test_create_shortlist():
    input_df = pl.DataFrame(
        [
            {
                "UID": "85028014",
                "Name": "- - ",
                "Club": "Man UFC",
                "Similarity": 100.0,
                "Chance Creation Rate": 0.003991130820399113,
                "Pass Completion Rate": 0.8423608516996638,
            },
            {
                "UID": "55070307",
                "Name": "- - ",
                "Club": "Wolves",
                "Similarity": 98.72976250063027,
                "Chance Creation Rate": 0.003627130939426913,
                "Pass Completion Rate": 0.881675727534378,
            },
            {
                "UID": "18077264",
                "Name": "- - ",
                "Club": "Leicester",
                "Similarity": 98.66699603523566,
                "Chance Creation Rate": 0.005043227665706052,
                "Pass Completion Rate": 0.8499693815064299,
            },
            {
                "UID": "91003875",
                "Name": "- - ",
                "Club": "Man City",
                "Similarity": 98.41038467380066,
                "Chance Creation Rate": 0.005832449628844115,
                "Pass Completion Rate": 0.9010989010989011,
            },
            {
                "UID": "85027988",
                "Name": "- - ",
                "Club": "Everton",
                "Similarity": 98.28159586046512,
                "Chance Creation Rate": 0.001257071024512885,
                "Pass Completion Rate": 0.8870922776693616,
            },
        ]
    )

    output_df = create_shortlist(input_df, "85028014")

    expected_df = pl.DataFrame(
        [
            {
                "UID": "85028014",
                "Name": "- - ",
                "Club": "Man UFC",
                "Similarity": 100.0,
                "Chance Creation Rate": 0.003991130820399113,
                "Pass Completion Rate": 0.8423608516996638,
            },
            {
                "UID": "18077264",
                "Name": "- - ",
                "Club": "Leicester",
                "Similarity": 98.66699603523566,
                "Chance Creation Rate": 0.005043227665706052,
                "Pass Completion Rate": 0.8499693815064299,
            },
            {
                "UID": "91003875",
                "Name": "- - ",
                "Club": "Man City",
                "Similarity": 98.41038467380066,
                "Chance Creation Rate": 0.005832449628844115,
                "Pass Completion Rate": 0.9010989010989011,
            },
        ]
    )
    assert output_df.equals(expected_df)


def test_create_shortlist():
    input_df = pl.DataFrame(
        [
            {
                "UID": "85028014",
                "Name": "- - ",
                "Club": "Man UFC",
                "Similarity": 100.0,
                "Chance Creation Rate": 0.003991130820399113,
                "Pass Completion Rate": 0.8423608516996638,
            },
            {
                "UID": "55070307",
                "Name": "- - ",
                "Club": "Wolves",
                "Similarity": 98.72976250063027,
                "Chance Creation Rate": 0.003627130939426913,
                "Pass Completion Rate": 0.881675727534378,
            },
            {
                "UID": "18077264",
                "Name": "- - ",
                "Club": "Leicester",
                "Similarity": 98.66699603523566,
                "Chance Creation Rate": 0.005043227665706052,
                "Pass Completion Rate": 0.8499693815064299,
            },
            {
                "UID": "91003875",
                "Name": "- - ",
                "Club": "Man City",
                "Similarity": 98.41038467380066,
                "Chance Creation Rate": 0.005832449628844115,
                "Pass Completion Rate": 0.9010989010989011,
            },
            {
                "UID": "85027988",
                "Name": "- - ",
                "Club": "Everton",
                "Similarity": 98.28159586046512,
                "Chance Creation Rate": 0.001257071024512885,
                "Pass Completion Rate": 0.8870922776693616,
            },
        ]
    )

    player_baseline = Baseline(chance_creation_rate=0.004, pass_completion_rate=0.9)

    output_df = filter_shortlist(input_df, player_baseline)

    expected_df = pl.DataFrame(
        [
            {
                "UID": "91003875",
                "Name": "- - ",
                "Club": "Man City",
                "Similarity": 98.41038467380066,
                "Chance Creation Rate": 0.005832449628844115,
                "Pass Completion Rate": 0.9010989010989011,
            },
        ]
    )

    assert output_df.equals(expected_df)


def test_find_player_baseline():
    transform_context = TransformContext(ColumnConfig(), "85028014")

    input_df = pl.DataFrame(
        [
            {
                "UID": "85028014",
                "Chance Creation Rate": 0.003991130820399113,
                "Pass Completion Rate": 0.8423608516996638,
            },
            {
                "UID": "55070307",
                "Chance Creation Rate": 0.003627130939426913,
                "Pass Completion Rate": 0.881675727534378,
            },
            {
                "UID": "18077264",
                "Chance Creation Rate": 0.005043227665706052,
                "Pass Completion Rate": 0.8499693815064299,
            },
            {
                "UID": "91003875",
                "Chance Creation Rate": 0.005832449628844115,
                "Pass Completion Rate": 0.9010989010989011,
            },
            {
                "UID": "85027988",
                "Chance Creation Rate": 0.001257071024512885,
                "Pass Completion Rate": 0.8870922776693616,
            },
        ]
    )

    player_baseline = find_player_baseline(input_df, transform_context)  # 85028014

    assert player_baseline.chance_creation_rate == 0.003991130820399113
    assert player_baseline.pass_completion_rate == 0.8423608516996638
