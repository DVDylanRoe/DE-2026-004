import polars as pl

def transform_columns_to_float(df):
    df = df.with_columns(
        (pl.col("Mins").str.replace(",", "").cast(pl.Float64).alias("Mins - float")),
        (pl.col("Hdrs A").cast(pl.Float64)).alias("Hdrs A - float"),
        (pl.col("Clear").cast(pl.Float64)).alias("Clear - float"),
        (pl.col("Cr A").cast(pl.Float64)).alias("Cr A - float"),
        (pl.col("Drb").cast(pl.Float64)).alias("Drb - float"),
        (pl.col("FA").cast(pl.Float64)).alias("FA - float"),
        (pl.col("Itc").cast(pl.Float64)).alias("Itc - float"),
        (pl.col("Pas A").str.replace(",", "").cast(pl.Float64)).alias("Pas A - float"),
        (pl.col("Ps C").str.replace(",", "").cast(pl.Float64)).alias("Ps C - float"),
        (pl.col("Shots").cast(pl.Float64)).alias("Shots - float"),
        (pl.col("Pens").cast(pl.Float64)).alias("Pens - float"),
        (pl.col("Tck W").cast(pl.Float64)).alias("Tck W - float"),
        (pl.col("Tck R").str.replace("%", "").cast(pl.Float64) / 100).alias(
            "Tck R - float"
        ),
        (pl.col("Yel").cast(pl.Float64)).alias("Yel - float"),
        (pl.col("Red").cast(pl.Float64)).alias("Red - float"),
        (pl.col("Fls").cast(pl.Float64)).alias("Fls - float"),
    )

    return df