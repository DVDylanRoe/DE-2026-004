import pytest
import polars as pl
from main import create_shortlist, find_player_baseline


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


def test_find_player_baseline():
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

    output_ccr, output_pcr = find_player_baseline(input_df, "85028014")

    assert output_ccr == 0.003991130820399113
    assert output_pcr == 0.8423608516996638
