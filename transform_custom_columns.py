import polars as pl

def transform_custom_columns(players_df):
    players_df = players_df.with_columns(
        (pl.col("Tck W - float") / pl.col("Tck R - float")).alias("Tck A")
    )

    players_df = players_df.with_columns(
        (pl.col("Shots - float") - pl.col("Pens - float")).alias("Non-Penalty Shots")
    )

    players_df = players_df.with_columns((pl.col("Mins - float") / 90).alias("90s"))

    return players_df