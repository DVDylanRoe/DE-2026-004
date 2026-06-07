import polars as pl
from dataclasses import dataclass
from similarity import compute_similarity
from config import TransformContext


def add_tackles_attempted(df):
    df = df.with_columns((pl.col("Tck W") / pl.col("Tck R")).alias("Tck A"))

    return df


def add_non_penalty_shots(df):
    df = df.with_columns((pl.col("Shots") - pl.col("Pens")).alias("Non Penalty Shots"))

    return df


def add_nineties_played(df):
    df = df.with_columns((pl.col("Mins") / 90).alias("90s"))

    return df


def transform_per90_columns(df: pl.DataFrame, context: TransformContext) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col(column) / pl.col("90s")).alias(f"{column} per 90")
            for column in context.config.per_ninety_source_columns
        ]
    )

    return df


def transform_Z_columns(df: pl.DataFrame, context: TransformContext) -> pl.DataFrame:

    df = df.with_columns(
        [
            ((pl.col(column) - pl.mean(column)) / pl.std(column)).alias(f"{column} Z")
            for column in context.config.per_ninety_columns
        ]
    )

    return df

def add_chance_creation_rate(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns((pl.col("CCC") / pl.col("Ps C")).alias("Chance Creation Rate"))

    return df


def add_pass_completion_rate(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("Ps C") / pl.col("Pas A")).alias("Pass Completion Rate"),
    )

    return df


def add_dervied_columns(df: pl.DataFrame, context: TransformContext):
    df = add_tackles_attempted(df)
    df = add_non_penalty_shots(df)
    df = add_nineties_played(df)
    df = transform_per90_columns(df, context)
    df = transform_Z_columns(df, context)
    df = compute_similarity(df, context)
    df = add_chance_creation_rate(df)
    df = add_pass_completion_rate(df)

    return df
