import polars as pl


def transform_Z_columns(players_df, PER_90_COLS):

    players_df = players_df.with_columns(
        [
            ((pl.col(col) - pl.mean(col)) / pl.std(col)).alias(col + " - Z")
            for col in PER_90_COLS
        ]
    )

    return players_df
