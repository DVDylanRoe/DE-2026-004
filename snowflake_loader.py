import snowflake.connector

def get_connection(credentials):
    conn = snowflake.connector.connect(
        account=credentials.get("account"),
        user=credentials.get("user"),
        authenticator=credentials.get("authenticator"),
        role=credentials.get("role"),
        warehouse=credentials.get("warehouse"),
        database=credentials.get("database"),
        schema=credentials.get("schema")
    )

    return conn