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


def load(players_df, transform_df, similar_df, shortlist_df):
    players_df.write_csv("data\\players-raw.csv")
    transform_df.write_csv("data\\replacing-pogba-1.1.csv")
    similar_df.write_csv("data\\replacing-pogba-1.3.csv")
    shortlist_df.select(
        [
            "UID",
            "Name",
            "Club",
            "Similarity",
            "Chance Creation Rate",
            "Pass Completion Rate",
        ]
    ).write_csv("data\\replacing-pogba-1.5.csv")
