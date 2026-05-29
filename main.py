from inputOutput import get_players_data, load
from config import ColumnConfig, TransformContext, Baseline
from pipeline import transform


def main():
    file_path = r"C:\Users\d_roe\Documents\VS Code Projects\Portfolio\DE-2026-004\players_20220522.html"
    column_config = ColumnConfig()
    transform_context = TransformContext(column_config, "85028014")

    players_df = get_players_data(file_path)

    transform_df, similar_df, shortlist_df = transform(players_df, transform_context)

    load(players_df, transform_df)


if __name__ == "__main__":
    main()
