from bs4 import BeautifulSoup
import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def read_html(file_path: str):
    with open(file_path, encoding="utf-8") as file:
        html = file.read()
        return html


def parse_html_table(html: str) -> tuple[list[str], list[list[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    headers = [header.text for header in table.select("tr th")]

    rows = [
        [data.text for data in row.find_all("td")] for row in table.select("tr + tr")
    ]
    return headers, rows


def get_players_data(file_path: str) -> pl.DataFrame:

    html = read_html(file_path)

    players_table_headers, players_table_rows = parse_html_table(html)

    players_df = pl.DataFrame(
        players_table_rows, schema=players_table_headers, orient="row"
    )

    return players_df


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


def add_tackles_attempted(df):
    df = df.with_columns((pl.col("Tck W") / pl.col("Tck R")).alias("Tck A"))

    return df


def add_non_penalty_shots(df):
    df = df.with_columns((pl.col("Shots") - pl.col("Pens")).alias("Non Penalty Shots"))

    return df


def add_nineties_played(df):
    df = df.with_columns((pl.col("Mins") / 90).alias("90s"))

    return df


def transform_per90_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col(column) / pl.col("90s")).alias(f"{column} per 90")
            for column in columns
        ]
    )

    return df


def transform_Z_columns(df: pl.DataFrame, per_ninety_columns: tuple) -> pl.DataFrame:

    df = df.with_columns(
        [
            ((pl.col(column) - pl.mean(column)) / pl.std(column)).alias(f"{column} Z")
            for column in per_ninety_columns
        ]
    )

    return df


def compute_similarity(df: pl.DataFrame, uid: str, columns: list[str]) -> pl.DataFrame:
    player = df.filter(pl.col("UID") == uid).select(columns)
    player_vector = player.to_numpy()

    feature_matrix = df.select(columns).to_numpy()

    cosine_similarity_scores = cosine_similarity(
        feature_matrix, player_vector
    ).flatten()
    cosine_similarity_scores_scaled = (cosine_similarity_scores * 50) + 50

    df = df.with_columns(
        [
            pl.Series("Similarity", cosine_similarity_scores_scaled.tolist()),
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


def find_similar_players(df: pl.DataFrame, threshold: int = 90) -> pl.DataFrame:

    similar_df = df.filter(pl.col("Similarity") >= threshold)

    return similar_df


def main():
    file_path = r"C:\Users\d_roe\Documents\VS Code Projects\Portfolio\DE-2026-004\players_20220522.html"

    players_df = get_players_data(file_path)

    players_df.write_csv("players-raw.csv")

    numeric_string_columns = ("Mins", "Pas A", "Ps C")
    numeric_columns = (
        "Mins",
        "Hdrs A",
        "Clear",
        "Cr A",
        "Drb",
        "FA",
        "Itc",
        "Pas A",
        "Ps C",
        "Shots",
        "Pens",
        "Tck W",
        "Yel",
        "Red",
        "Fls",
        "CCC",
    )
    percentage_columns = ["Tck R"]

    players_df = clean_numeric_string_columns(players_df, numeric_string_columns)
    players_df = cast_numeric_columns(players_df, numeric_columns)
    players_df = convert_percentage_columns(players_df, percentage_columns)

    players_df = add_tackles_attempted(players_df)
    players_df = add_non_penalty_shots(players_df)
    players_df = add_nineties_played(players_df)

    per_ninety_source_columns = (
        "Hdrs A",
        "Clear",
        "Cr A",
        "Drb",
        "FA",
        "Itc",
        "Pas A",
        "Ps C",
        "Non Penalty Shots",
        "Tck A",
        "Yel",
        "Red",
        "Fls",
    )

    players_df = transform_per90_columns(players_df, per_ninety_source_columns)

    per_ninety_columns = (
        "Hdrs A per 90",
        "Clear per 90",
        "Cr A per 90",
        "Drb per 90",
        "Itc per 90",
        "Pas A per 90",
        "Non Penalty Shots per 90",
        "Tck A per 90",
    )

    players_df = transform_Z_columns(players_df, per_ninety_columns)

    zscore_feature_columns = [column + " Z" for column in per_ninety_columns]

    players_df = compute_similarity(players_df, "85028014", zscore_feature_columns)

    players_df.write_csv("replacing-pogba-1.1.csv")

    players_df = add_chance_creation_rate(players_df)
    players_df = add_pass_completion_rate(players_df)

    players_df.write_csv("replacing-pogba-1.1.csv")

    # filter for similar players#

    similar_df = find_similar_players(players_df)

    # filter for similar players#

    similar_df.write_csv("replacing-pogba-1.3.csv")

    pogba_row = similar_df.filter(pl.col("UID") == "85028014").select(
        ["Chance Creation Rate", "Pass Completion Rate"]
    )

    pogba_chance_creation_rate = pogba_row["Chance Creation Rate"][0]
    pogba_pass_completion_rate = pogba_row["Pass Completion Rate"][0]

    shortlist_df = players_df.filter(
        (pl.col("Similarity") >= 90)
        & (pl.col("Chance Creation Rate") >= pogba_chance_creation_rate)
        & (pl.col("Pass Completion Rate") >= pogba_pass_completion_rate)
    )

    shortlist_df.select(
        [
            "UID",
            "Name",
            "Club",
            "Similarity",
            "Chance Creation Rate",
            "Pass Completion Rate",
        ]
    ).write_csv("replacing-pogba-1.5.csv")

    # create shortlist#


if __name__ == "__main__":
    main()
