from config import LoadTarget
import snowflake.connector
from pathlib import Path


class SnowflakeClient:
    def __init__(self, credentials):
        self.connection = self._get_connection(credentials)

    def _get_connection(self, credentials):
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

    def _generate_put_command(self, target: LoadTarget) -> str:
        p = Path(target.csv_path_str)
        source_uri = "file://" + p.as_posix()

        put_cmd = f"""
PUT '{source_uri}'
@%{target.table}
AUTO_COMPRESS=TRUE
OVERWRITE=TRUE;
"""
        return put_cmd

    def _stage_data(self, put_command):
        cur = self.connection.cursor()
        cur.execute(put_command)
        cur.close()

    def _truncate_table(self, target):
        cur = self.connection.cursor()
        cur.execute(f"TRUNCATE {target.table}")
        cur.close()

    def _generate_copy_command(self, target):
        copy_cmd = f"""
COPY INTO {target.table}
FROM @%{target.table}
FILE_FORMAT = (
    TYPE = CSV
    FIELD_DELIMITER = ','
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
)
"""

        return copy_cmd

    def _copy_data_into_table(self, copy_command):
        cur = self.connection.cursor()
        cur.execute(copy_command)
        cur.close()

    def load(self, targets):
        for target in targets:

            if target.table:
                put_cmd = self._generate_put_command(target)
                self._stage_data(put_cmd)

                self._truncate_table(target)

                copy_cmd = self._generate_copy_command(target)
                self._copy_data_into_table(copy_cmd)

        self.connection.close()


def load(players_df, transform_df):
    players_df.write_csv("data\\players_raw.csv")
    transform_df.write_csv("data\\players_transformed.csv")


def write_targets(targets):
    for target in targets:
        target.df.write_csv(target.csv_path)
