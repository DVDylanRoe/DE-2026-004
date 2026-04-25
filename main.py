from bs4 import BeautifulSoup
import polars as pl


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


def transform_per90_columns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("Hdrs A") / pl.col("90s")).alias("Hdrs A per 90"),
        (pl.col("Clear") / pl.col("90s")).alias("Clear per 90"),
        (pl.col("Cr A") / pl.col("90s")).alias("Cr A per 90"),
        (pl.col("Drb") / pl.col("90s")).alias("Drb per 90"),
        (pl.col("FA") / pl.col("90s")).alias("FA per 90"),
        (pl.col("Itc") / pl.col("90s")).alias("Itc per 90"),
        (pl.col("Pas A") / pl.col("90s")).alias("Pas A per 90"),
        (pl.col("Ps C") / pl.col("90s")).alias("Ps C per 90"),
        (pl.col("Non Penalty Shots") / pl.col("90s")).alias("Non Penalty Shots per 90"),
        (pl.col("Tck A") / pl.col("90s")).alias("Tck A per 90"),
        (pl.col("Yel") / pl.col("90s")).alias("Yel per 90"),
        (pl.col("Red") / pl.col("90s")).alias("Red per 90"),
        (pl.col("Fls") / pl.col("90s")).alias("Fls per 90"),
    )

    return df


def transform_Z_columns(df: pl.DataFrame, per_ninety_columns: tuple) -> pl.DataFrame:

    df = df.with_columns(
        [
            ((pl.col(column) - pl.mean(column)) / pl.std(column)).alias(column + " Z")
            for column in per_ninety_columns
        ]
    )

    return df


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

    players_df = transform_per90_columns(players_df)

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

    # calculate cos sim#

    import numpy as np

    pogba = players_df.filter(pl.col("UID") == "85028014").select(
        zscore_feature_columns
    )
    pogba_vector = pogba.to_numpy()

    feature_matrix = players_df.select(zscore_feature_columns).to_numpy()

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarity_scores = cosine_similarity(feature_matrix, pogba_vector).flatten()
    cosine_similarity_scores_scaled = (
        cosine_similarity_scores * 50
    ) + 50  # scale to 0–100

    players_df = players_df.with_columns(
        [
            pl.Series("Pogba Similarity", cosine_similarity_scores_scaled.tolist()),
        ]
    )
    # calculate cos sim#

    players_df.write_csv("replacing-pogba-1.1.csv")

    # calculate ccr and pcr#

    players_df = players_df.with_columns(
        (pl.col("CCC") / pl.col("Ps C")).alias("Chance Creation Rate"),
        (pl.col("Ps C") / pl.col("Pas A")).alias("Pass Completion Rate"),
    )

    # calculate ccr and pcr#

    players_df.write_csv("replacing-pogba-1.1.csv")

    # filter for similar players#

    midfielders_df = players_df.filter(pl.col("Pogba Similarity") >= 90)

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
        (pl.col("Pogba Similarity") >= 90)
        & (pl.col("Chance Creation Rate") >= pogba_chance_creation_rate)
        & (pl.col("Pass Completion Rate") >= pogba_pass_completion_rate)
    )

    shortlist_df.select(
        [
            "UID",
            "Name",
            "Club",
            "Pogba Similarity",
            "Chance Creation Rate",
            "Pass Completion Rate",
        ]
    ).write_csv("replacing-pogba-1.5.csv")

    # create shortlist#


if __name__ == "__main__":
    main()
