import snowflake.connector
from pathlib import Path


def get_connection(credentials):
    conn = snowflake.connector.connect(
        account=credentials.account,
        user=credentials.user,
        authenticator=credentials.authenticator,
        role=credentials.role,
        warehouse=credentials.warehouse,
        database=credentials.database,
        schema=credentials.schema,
    )

    return conn


def generate_put_command(source_file_path: str, target_table: str) -> str:
    p = Path(source_file_path)
    source_uri = "file://" + p.as_posix()

    put_cmd = f"""
        PUT '{source_uri}'
        @%{target_table}
        AUTO_COMPRESS=TRUE
        OVERWRITE=TRUE;
        """

    return put_cmd


def stage_data(connection, put_command):
    cur = connection.cursor()
    cur.execute(put_command)
    cur.close()

    return "success"


def copy_data_into_table(connection, copy_command):
    cur = connection.cursor()

    cur.execute(copy_command)
    cur.close()
