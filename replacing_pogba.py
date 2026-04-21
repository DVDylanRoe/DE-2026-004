from bs4 import BeautifulSoup
import polars as pl

from get_players_data import get_players_data

from transform_columns_to_float import transform_columns_to_float
from transform_custom_columns import transform_custom_columns
from transform_per90_columns import transform_per90_columns

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

    players_df = transform_columns_to_float(players_df)

    players_df = transform_custom_columns(players_df)

    players_df = transform_per90_columns(players_df)

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
