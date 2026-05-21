import polars as pl
from dataclasses import dataclass


def clean_numeric_string_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    df = df.with_columns([pl.col(column).str.replace(",", "") for column in columns])

    return df


def cast_numeric_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    df = df.with_columns([pl.col(column).cast(pl.Float64) for column in columns])

    return df


def convert_percentage_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col(column).str.replace("%", "").cast(pl.Float64) / 100)
            for column in columns
        ]
    )

    return df


def clean_data(df: pl.DataFrame, Config: dataclass):
    df = clean_numeric_string_columns(df, Config.numeric_string_columns)
    df = cast_numeric_columns(df, Config.numeric_columns)
    df = convert_percentage_columns(df, Config.percentage_columns)

    return df
