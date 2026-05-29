from bs4 import BeautifulSoup
import polars as pl
from snowflake_loader import (
    get_connection,
    generate_put_command,
    stage_data,
    generate_copy_command,
    copy_data_into_table,
)
from config import LoadTarget


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


def load(players_df, transform_df):
    players_df.write_csv("data\\players_raw.csv")
    transform_df.write_csv("data\\players_transformed.csv")


def load_to_snowflake(credentials, targets: list[LoadTarget]):
    conn = None
    if credentials:
        conn = get_connection(credentials)

    for target in targets:
        target.df.write_csv(target.csv_path_str)

        if target.table:
            put_cmd = generate_put_command(target.csv_path_str, target.table)
            stage_data(conn, put_cmd)

            copy_cmd = generate_copy_command(target.table)
            copy_data_into_table(conn, copy_cmd)

    if conn:
        conn.close()
