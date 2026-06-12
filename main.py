from html_reader import get_players_data
from loaders import SnowflakeClient, write_targets
from config import (
    ColumnConfig,
    TransformContext,
    SnowflakeCredentials,
    LoadTarget,
    load_yaml,
    resolve_uid,
)
from pipeline import transform
from cli import build_parser

from dotenv import load_dotenv
from pathlib import Path
import os


def main(load_sf=True):
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)

    if "PYTEST_CURRENT_TEST" in os.environ:
        fixture_html = Path(__file__).parent / "tests" / "fixtures" / "players.html"
        config["input_html"] = str(fixture_html)

    uid = str(resolve_uid(config, args))

    column_config = ColumnConfig()
    transform_context = TransformContext(column_config, uid)

    players_df = get_players_data(Path(config["input_html"]).resolve())

    pipeline_result = transform(players_df, transform_context)

    targets = [
        LoadTarget(None, pipeline_result.raw, Path(config["output_raw"]).resolve()),
        LoadTarget(
            "players" if load_sf else None,
            pipeline_result.transformed,
            Path(config["output_transformed"]).resolve(),
        ),
    ]

    write_targets(targets)

    if load_sf:
        load_dotenv()
        credentials = SnowflakeCredentials()
        snowflake_client = SnowflakeClient(credentials)
        snowflake_client.load(targets)


if __name__ == "__main__":
    main()
