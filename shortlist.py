import polars as pl
from dataclasses import dataclass
from config import Baseline


def find_player_baseline(df: pl.DataFrame, context: dataclass) -> tuple[float, float]:
    player_stats = df.filter(pl.col("UID") == context.uid).select(
        ["Chance Creation Rate", "Pass Completion Rate"]
    )

    player_chance_creation_rate = player_stats["Chance Creation Rate"][0]
    player_pass_completion_rate = player_stats["Pass Completion Rate"][0]

    player_baseline = Baseline(
        chance_creation_rate=player_chance_creation_rate,
        pass_completion_rate=player_pass_completion_rate,
    )

    return player_baseline


def filter_shortlist(df: pl.DataFrame, baseline: dataclass):
    filtered_df = df.filter(
        (pl.col("Chance Creation Rate") >= baseline.chance_creation_rate)
        & (pl.col("Pass Completion Rate") >= baseline.pass_completion_rate)
    )

    return filtered_df


def create_shortlist(df: pl.DataFrame, context: dataclass) -> pl.DataFrame:
    player_baseline = find_player_baseline(df, context)

    shortlist_df = filter_shortlist(df, player_baseline)

    return shortlist_df
