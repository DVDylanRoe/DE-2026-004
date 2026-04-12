from bs4 import BeautifulSoup
import polars as pl


def get_players_data(file_path: str) -> pl.DataFrame:

    with open(file_path, encoding="utf-8") as f:
        html = f.read()

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
