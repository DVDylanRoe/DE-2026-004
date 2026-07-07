from core.features import add_features
import polars as pl
from config import TransformContext, ColumnConfig
from polars.testing import assert_frame_equal


def test_add_features():
    input_df = pl.DataFrame(
        [
            {
                "UID": "719601",
                "Tck W": 33.0,
                "Tck R": 0.6900000000000001,
                "Shots": 261.0,
                "Pens": 15.0,
                "Mins": 4064.0,
                "Pas A": 1594.0,
                "Ps C": 1362.0,
                "CCC": 13.0,
            },
            {
                "UID": "29179241",
                "Tck W": 37.0,
                "Tck R": 0.79,
                "Shots": 289.0,
                "Pens": 9.0,
                "Mins": 4025.0,
                "Pas A": 1704.0,
                "Ps C": 1429.0,
                "CCC": 24.0,
            },
            {
                "UID": "62182055",
                "Tck W": 19.0,
                "Tck R": 0.61,
                "Shots": 195.0,
                "Pens": 10.0,
                "Mins": 3950.0,
                "Pas A": 1682.0,
                "Ps C": 1395.0,
                "CCC": 8.0,
            },
        ]
    )

    context = TransformContext(
        config=ColumnConfig(
            per_ninety_source_columns=[
                "Non Penalty Shots",
                "Tck A",
            ],
            per_ninety_columns=[
                "Non Penalty Shots per 90",
                "Tck A per 90",
            ],
        ),
        uid=None,
    )

    output_df = add_features(input_df, context)

    expected_df = pl.DataFrame(
        [
            {
                "UID": "719601",
                "Tck W": 33.0,
                "Tck R": 0.6900000000000001,
                "Shots": 261.0,
                "Pens": 15.0,
                "Mins": 4064.0,
                "Pas A": 1594.0,
                "Ps C": 1362.0,
                "CCC": 13.0,
                "Tck A": 47.82608695652173,
                "Non Penalty Shots": 246.0,
                "90s": 45.15555555555556,
                "Non Penalty Shots per 90": 5.447834645669291,
                "Tck A per 90": 1.059140705237932,
                "Non Penalty Shots per 90 Z": 0.13579772950217103,
                "Tck A per 90 Z": 0.6070503734928266,
                "Chance Creation Rate": 0.009544787077826725,
                "Pass Completion Rate": 0.8544542032622334,
            },
            {
                "UID": "29179241",
                "Tck W": 37.0,
                "Tck R": 0.79,
                "Shots": 289.0,
                "Pens": 9.0,
                "Mins": 4025.0,
                "Pas A": 1704.0,
                "Ps C": 1429.0,
                "CCC": 24.0,
                "Tck A": 46.83544303797468,
                "Non Penalty Shots": 280.0,
                "90s": 44.72222222222222,
                "Non Penalty Shots per 90": 6.260869565217392,
                "Tck A per 90": 1.04725214246403,
                "Non Penalty Shots per 90 Z": 0.9251616734320308,
                "Tck A per 90 Z": 0.5471318434553384,
                "Chance Creation Rate": 0.016794961511546535,
                "Pass Completion Rate": 0.8386150234741784,
            },
            {
                "UID": "62182055",
                "Tck W": 19.0,
                "Tck R": 0.61,
                "Shots": 195.0,
                "Pens": 10.0,
                "Mins": 3950.0,
                "Pas A": 1682.0,
                "Ps C": 1395.0,
                "CCC": 8.0,
                "Tck A": 31.14754098360656,
                "Non Penalty Shots": 185.0,
                "90s": 43.88888888888889,
                "Non Penalty Shots per 90": 4.215189873417721,
                "Tck A per 90": 0.7096908072214153,
                "Non Penalty Shots per 90 Z": -1.0609594029342027,
                "Tck A per 90 Z": -1.1541822169481657,
                "Chance Creation Rate": 0.005734767025089606,
                "Pass Completion Rate": 0.8293697978596909,
            },
        ]
    )

    assert_frame_equal(output_df, expected_df, check_exact=False)
