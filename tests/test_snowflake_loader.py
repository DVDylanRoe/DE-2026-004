import snowflake_loader
from unittest.mock import patch

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

    snowflake_loader.get_connection(credentials)

    mock_connect.assert_called_once_with(
        account="constant",
        user="rocket",
        authenticator="helmet",
        role="multiply",
        warehouse="please",
        database="embarrassment",
        schema="offspring",
    )
