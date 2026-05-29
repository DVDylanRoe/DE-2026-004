from inputOutput import get_players_data, load_to_snowflake
from config import (
    ColumnConfig,
    TransformContext,
    Baseline,
    SnowflakeCredentials,
    LoadTarget,
)
from pipeline import transform
from dotenv import load_dotenv
from pathlib import Path


def main(load_sf=True):
    if load_sf:
        load_dotenv()
        credentials = SnowflakeCredentials()
    else:
        credentials = None

    file_path = r"C:\Users\d_roe\Documents\VS Code Projects\Portfolio\DE-2026-004\players_20220522.html"
    column_config = ColumnConfig()
    transform_context = TransformContext(column_config, "85028014")

    players_df = get_players_data(file_path)

    transform_df, similar_df, shortlist_df = transform(players_df, transform_context)

    targets = [
        LoadTarget(None, players_df, Path("data/players_raw.csv").resolve()),
        LoadTarget(
            "players" if load_sf else None,
            transform_df,
            Path("data/players_transformed.csv").resolve(),
        ),
    ]

    load_to_snowflake(credentials, targets)


if __name__ == "__main__":
    main()
