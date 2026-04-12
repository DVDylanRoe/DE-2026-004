from bs4 import BeautifulSoup
import polars as pl
from get_players_data import get_players_data

def main():
    FILE_PATH = r"C:\Users\d_roe\Documents\VS Code Projects\Portfolio\DE-2026-004\players_20220522.html"

    players_df = get_players_data(FILE_PATH)

    players_df.write_csv("players-raw.csv")

    FEATURES = [
        "Mins",
        "Hdrs A",
        "Clear",
        "Cr A",
        "Drb",
        "FA",
        "Off",
        "Itc",
        "Pas A",
        "Shots",
        "Pens",
        "Tck W",
        "Tck R",
        "Yel",
        "Red",
        "Fls",
    ]

    players_df = players_df.with_columns(
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

    players_df = players_df.with_columns(
        (pl.col("Tck W - float") / pl.col("Tck R - float")).alias("Tck A")
    )

    players_df = players_df.with_columns(
        (pl.col("Shots - float") - pl.col("Pens - float")).alias("Non-Penalty Shots")
    )

    players_df = players_df.with_columns((pl.col("Mins - float") / 90).alias("90s"))

    players_df = players_df.with_columns(
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

    PER_90_COLS = [
        "Hdrs A per 90",
        "Clear per 90",
        "Cr A per 90",
        "Drb per 90",
        "Itc per 90",
        "Pas A per 90",
        "Non-Penalty Shots per 90",
        "Tck A per 90",
    ]

    players_df = players_df.with_columns(
        [
            ((pl.col(col) - pl.mean(col)) / pl.std(col)).alias(col + " - Z")
            for col in PER_90_COLS
        ]
    )

    FEATURES = [col + " - Z" for col in PER_90_COLS]

    import numpy as np

    pogba = players_df.filter(pl.col("UID") == "85028014").select(FEATURES)
    pogba_vec = pogba.to_numpy()

    X = players_df.select(FEATURES).to_numpy()

    from sklearn.metrics.pairwise import cosine_similarity

    cos_sim = cosine_similarity(X, pogba_vec).flatten()
    cos_sim_scaled = (cos_sim * 50) + 50  # scale to 0–100

    players_df = players_df.with_columns(
        [
            pl.Series("Pogba Similarity (Cosine)", cos_sim_scaled.tolist()),
        ]
    )

    players_df.write_csv("replacing-pogba-1.1.csv")


    players_df = players_df.with_columns(
        (pl.col("CCC").cast(pl.Float64)).alias("CCC - float"),
    )

    players_df = players_df.with_columns(
        (pl.col("CCC - float") / pl.col("Ps C - float")).alias("Chance Creation Rate"),
        (pl.col("Ps C - float") / pl.col("Pas A - float")).alias("Pass Completion Rate"),
    )

    players_df.write_csv("replacing-pogba-1.1.csv")

    midfielders_df = players_df.filter(pl.col("Pogba Similarity (Cosine)") >= 90)

    midfielders_df.write_csv("replacing-pogba-1.3.csv")

    ccr = midfielders_df.get_column("Chance Creation Rate").cast(pl.Float64).to_list()
    pcr = midfielders_df.get_column("Pass Completion Rate").cast(pl.Float64).to_list()


    ccr_mean = midfielders_df.select(pl.mean("Chance Creation Rate")).row(0)
    pcc_mean = midfielders_df.select(pl.mean("Pass Completion Rate")).row(0)

    import numpy as np

    pogba_row = midfielders_df.filter(pl.col("UID") == "85028014").select(
        ["Chance Creation Rate", "Pass Completion Rate"]
    )
    pogba_ccr = pogba_row["Chance Creation Rate"][0]
    pogba_pcr = pogba_row["Pass Completion Rate"][0]

    ccr = midfielders_df.get_column("Chance Creation Rate").cast(pl.Float64).to_list()
    pcr = midfielders_df.get_column("Pass Completion Rate").cast(pl.Float64).to_list()

    ccr_mean = midfielders_df.select(pl.mean("Chance Creation Rate")).row(0)
    pcr_mean = midfielders_df.select(pl.mean("Pass Completion Rate")).row(0)


    pogba_row = midfielders_df.filter(pl.col("UID") == "85028014").select(
        ["Chance Creation Rate", "Pass Completion Rate"]
    )

    pogba_ccr = pogba_row["Chance Creation Rate"][0]
    pogba_pcr = pogba_row["Pass Completion Rate"][0]

    shortlist_df = players_df.filter(
        (pl.col("Pogba Similarity (Cosine)") >= 90)
        & (pl.col("Chance Creation Rate") >= pogba_ccr)
        & (pl.col("Pass Completion Rate") >= pogba_pcr)
    )

    shortlist_df.select(
        [
            "UID",
            "Name",
            "Club",
            "Pogba Similarity (Cosine)",
            "Chance Creation Rate",
            "Pass Completion Rate",
        ]
    ).write_csv("replacing-pogba-1.5.csv")

if __name__ == "__main__":
    main()
