from main import transform_custom_columns
import polars as pl

def test_transform_custom_columns():
    input_df = pl.DataFrame(
        [
            {
                "UID": "719601",
                "Tck W": 33.0,
                "Tck R": 0.6900000000000001,
                "Shots": 261.0,
                "Pens": 15.0,
                "Mins": 4064.0,
            },
            {
                "UID": "29179241",
                "Tck W": 37.0,
                "Tck R": 0.79,
                "Shots": 289.0,
                "Pens": 9.0,
                "Mins": 4025.0,
            },
            {
                "UID": "62182055",
                "Tck W": 19.0,
                "Tck R": 0.61,
                "Shots": 195.0,
                "Pens": 10.0,
                "Mins": 3950.0,
            },
        ]
    )

    output_df = transform_custom_columns(input_df)

    expected_df = pl.DataFrame(
        [
            {
                "UID": "719601",
                "Tck W": 33.0,
                "Tck R": 0.6900000000000001,
                "Shots": 261.0,
                "Pens": 15.0,
                "Mins": 4064.0,
                "Tck A": 47.826086956521735,
                "Non Penalty Shots": 246.0,
                "90s": 45.15555555555556,
            },
            {
                "UID": "29179241",
                "Tck W": 37.0,
                "Tck R": 0.79,
                "Shots": 289.0,
                "Pens": 9.0,
                "Mins": 4025.0,
                "Tck A": 46.835443037974684,
                "Non Penalty Shots": 280.0,
                "90s": 44.72222222222222,
            },
            {
                "UID": "62182055",
                "Tck W": 19.0,
                "Tck R": 0.61,
                "Shots": 195.0,
                "Pens": 10.0,
                "Mins": 3950.0,
                "Tck A": 31.147540983606557,
                "Non Penalty Shots": 185.0,
                "90s": 43.88888888888889,
            },
        ]
    )

    assert output_df.equals(expected_df)
