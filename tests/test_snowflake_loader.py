from unittest.mock import patch, MagicMock
import pytest
from config import SnowflakeCredentials, LoadTarget
from external.loaders import SnowflakeClient


@pytest.fixture
def mock_connect():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@patch("external.loaders.snowflake.connector.connect")
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

    snowflake_client = SnowflakeClient(credentials)

    mock_connect.assert_called_once_with(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )


@patch("external.loaders.snowflake.connector.connect")
def test_stage_data(mock_connect):

    credentials = SnowflakeCredentials(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )

    snowflake_client = SnowflakeClient(credentials)

    put_cmd = "PUT file:///tmp/data/file* @%testtable"

    snowflake_client._stage_data(put_cmd)

    mock_cursor = mock_connect.return_value.cursor.return_value

    mock_cursor.execute.assert_called_once_with(put_cmd)


@patch("external.loaders.snowflake.connector.connect")
def test_copy_data_into_table(mock_connect):

    credentials = SnowflakeCredentials(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )

    snowflake_client = SnowflakeClient(credentials)

    copy_cmd = "COPY INTO testtable"

    snowflake_client._copy_data_into_table(copy_cmd)

    mock_cursor = mock_connect.return_value.cursor.return_value

    mock_cursor.execute.assert_called_once_with(copy_cmd)


@patch("external.loaders.snowflake.connector.connect")
def test_generate_put_command(mock_connect):

    credentials = SnowflakeCredentials(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )

    snowflake_client = SnowflakeClient(credentials)

    source_file_path = "C:\path\squid"
    target_table = "squid"

    target = LoadTarget(target_table, None, source_file_path)

    put_cmd = snowflake_client._generate_put_command(target)

    expected_put_cmd = """
PUT 'file://C:/path/squid'
@%squid
AUTO_COMPRESS=TRUE
OVERWRITE=TRUE;
"""

    assert put_cmd == expected_put_cmd


@patch("external.loaders.snowflake.connector.connect")
def test_generate_copy_command(mock_connect):

    credentials = SnowflakeCredentials(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )

    snowflake_client = SnowflakeClient(credentials)

    target_table = "squid"

    target = LoadTarget(target_table, None, None)

    copy_cmd = snowflake_client._generate_copy_command(target)

    expected_copy_cmd = """
COPY INTO squid
FROM @%squid
FILE_FORMAT = (
    TYPE = CSV
    FIELD_DELIMITER = ','
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
)
"""

    assert copy_cmd == expected_copy_cmd
