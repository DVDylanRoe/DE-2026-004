from bs4 import BeautifulSoup
import polars as pl


def get_players_data(file_path: str) -> pl.DataFrame:

    with open(file_path, encoding="utf-8") as file:
        html = file.read()

    soup = BeautifulSoup(html, "html.parser")
    players_table = soup.find("table")

    players_table_headers = [th.text for th in players_table.select("tr th")]

    players_table_rows = [
        [td.text for td in row.find_all("td")]
        for row in players_table.select("tr + tr")
    ]

    players_df = pl.DataFrame(
        players_table_rows, schema=players_table_headers, orient="row"
    )

    return players_df


def transform_columns_to_float(df: pl.DataFrame) -> pl.DataFrame:
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


def transform_custom_columns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("Tck W - float") / pl.col("Tck R - float")).alias("Tck A")
    )

    df = df.with_columns(
        (pl.col("Shots - float") - pl.col("Pens - float")).alias("Non-Penalty Shots")
    )

    df = df.with_columns((pl.col("Mins - float") / 90).alias("90s"))

    return df


def transform_per90_columns(df: pl.DataFrame) -> pl.DataFrame:
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


def transform_Z_columns(df: pl.DataFrame, PER_NINETY_COLUMNS: tuple) -> pl.DataFrame:

    df = df.with_columns(
        [
            ((pl.col(column) - pl.mean(column)) / pl.std(column)).alias(column + " - Z")
            for column in PER_NINETY_COLUMNS
        ]
    )

    return df


def main():
    FILE_PATH = r"C:\Users\d_roe\Documents\VS Code Projects\Portfolio\DE-2026-004\players_20220522.html"

    players_df = get_players_data(FILE_PATH)

    players_df.write_csv("players-raw.csv")

    players_df = transform_columns_to_float(players_df)

    players_df = transform_custom_columns(players_df)

    players_df = transform_per90_columns(players_df)

    PER_NINETY_COLUMNS = (
        "Hdrs A per 90",
        "Clear per 90",
        "Cr A per 90",
        "Drb per 90",
        "Itc per 90",
        "Pas A per 90",
        "Non-Penalty Shots per 90",
        "Tck A per 90",
    )

    players_df = transform_Z_columns(players_df, PER_NINETY_COLUMNS)

    zscore_feature_columns = [column + " - Z" for column in PER_NINETY_COLUMNS]

    # calculate cos sim#

    import numpy as np

    pogba = players_df.filter(pl.col("UID") == "85028014").select(zscore_feature_columns)
    pogba_vector = pogba.to_numpy()

    feature_matrix = players_df.select(zscore_feature_columns).to_numpy()

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarity_scores = cosine_similarity(feature_matrix, pogba_vector).flatten()
    cosine_similarity_scores_scaled = (cosine_similarity_scores * 50) + 50  # scale to 0–100

    players_df = players_df.with_columns(
        [
            pl.Series("Pogba Similarity (Cosine)", cosine_similarity_scores_scaled.tolist()),
        ]
    )
    # calculate cos sim#

    players_df.write_csv("replacing-pogba-1.1.csv")

    # calculate ccr and pcr#

    players_df = players_df.with_columns(
        (pl.col("CCC").cast(pl.Float64)).alias("CCC - float"),
    )

    players_df = players_df.with_columns(
        (pl.col("CCC - float") / pl.col("Ps C - float")).alias("Chance Creation Rate"),
        (pl.col("Ps C - float") / pl.col("Pas A - float")).alias(
            "Pass Completion Rate"
        ),
    )

    # calculate ccr and pcr#

    players_df.write_csv("replacing-pogba-1.1.csv")

    # filter for similar players#

    midfielders_df = players_df.filter(pl.col("Pogba Similarity (Cosine)") >= 90)

    # filter for similar players#

    midfielders_df.write_csv("replacing-pogba-1.3.csv")

    import numpy as np

    # create shortlist#

    pogba_row = midfielders_df.filter(pl.col("UID") == "85028014").select(
        ["Chance Creation Rate", "Pass Completion Rate"]
    )

    pogba_chance_creation_rate = pogba_row["Chance Creation Rate"][0]
    pogba_pass_completion_rate = pogba_row["Pass Completion Rate"][0]

    shortlist_df = players_df.filter(
        (pl.col("Pogba Similarity (Cosine)") >= 90)
        & (pl.col("Chance Creation Rate") >= pogba_chance_creation_rate)
        & (pl.col("Pass Completion Rate") >= pogba_pass_completion_rate)
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

    # create shortlist#


if __name__ == "__main__":
    main()
