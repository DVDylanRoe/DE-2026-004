import polars as pl


def transform_per90_columns(df):
    df = df.with_columns(
        (pl.col("Hdrs A - float") / pl.col("90s")).alias("Hdrs A per 90"),
        (pl.col("Clear - float") / pl.col("90s")).alias("Clear per 90"),
        (pl.col("Cr A - float") / pl.col("90s")).alias("Cr A per 90"),
        (pl.col("Drb - float") / pl.col("90s")).alias("Drb per 90"),
        (pl.col("FA - float") / pl.col("90s")).alias("FA per 90"),
        (pl.col("Itc - float") / pl.col("90s")).alias("Itc per 90"),
        (pl.col("Pas A - float") / pl.col("90s")).alias("Pas A per 90"),
        (pl.col("Ps C - float") / pl.col("90s")).alias("Ps C per 90"),
        (pl.col("Non-Penalty Shots") / pl.col("90s")).alias("Non-Penalty Shots per 90"),
        (pl.col("Tck A") / pl.col("90s")).alias("Tck A per 90"),
        (pl.col("Yel - float") / pl.col("90s")).alias("Yel per 90"),
        (pl.col("Red - float") / pl.col("90s")).alias("Red per 90"),
        (pl.col("Fls - float") / pl.col("90s")).alias("Fls per 90"),
    )

    return df
