from snowflake_loader import get_connection, land_data
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture
def mock_connect():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@patch("snowflake_loader.snowflake.connector.connect")
def test_get_connection(mock_connect):
    credentials = {
        "account": "constant",
        "user": "rocket",
        "authenticator": "helmet",
        "role": "multiply",
        "warehouse": "please",
        "database": "embarrassment",
        "schema": "offspring",
    }

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


def test_land_data(mock_connect):
    put_cmd = "PUT file:///tmp/data/file* @%testtable"

    land_data(mock_connect, put_cmd)

    mock_cursor = mock_connect.cursor.return_value

    mock_cursor.execute.assert_called_once_with(put_cmd)
