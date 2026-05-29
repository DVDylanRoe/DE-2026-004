from snowflake_loader import (
    get_connection,
    stage_data,
    copy_data_into_table,
    generate_put_command,
)
from unittest.mock import patch, MagicMock
import pytest
from config import SnowflakeCredentials


@pytest.fixture
def mock_connect():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@patch("snowflake_loader.snowflake.connector.connect")
def test_get_connection(mock_connect):
    credentials = SnowflakeCredentials(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )

    get_connection(credentials)

    mock_connect.assert_called_once_with(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )


def test_stage_data(mock_connect):
    put_cmd = "PUT file:///tmp/data/file* @%testtable"

    stage_data(mock_connect, put_cmd)

    mock_cursor = mock_connect.cursor.return_value

    mock_cursor.execute.assert_called_once_with(put_cmd)


def test_copy_data_into_table(mock_connect):
    copy_cmd = "COPY INTO testtable"

    copy_data_into_table(mock_connect, copy_cmd)

    mock_cursor = mock_connect.cursor.return_value

    mock_cursor.execute.assert_called_once_with(copy_cmd)


def test_generate_put_command():
    source_file_path = "C:\path\squid"
    target_table = "squid"

    put_cmd = generate_put_command(source_file_path, target_table)

    expected_put_cmd = """
        PUT 'file://C:/path/squid'
        @%squid
        AUTO_COMPRESS=TRUE
        OVERWRITE=TRUE;
        """

    assert put_cmd == expected_put_cmd
