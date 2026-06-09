from snowflake_loader import (
    get_connection,
    generate_put_command,
    stage_data,
    generate_copy_command,
    copy_data_into_table,
    truncate_table,
)
from config import LoadTarget


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

            truncate_table(conn, target.table)

            copy_cmd = generate_copy_command(target.table)
            copy_data_into_table(conn, copy_cmd)

    if conn:
        conn.close()


def write_targets(targets):
    for target in targets:
        target.df.write_csv(target.csv_path)
