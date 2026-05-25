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

def land_data(connection, put_command):
    # put_cmd = """
    # PUT 'file://C:/Users/d_roe/Documents/VS Code Projects/Portfolio/DE-2026-004/tests/golden/replacing-pogba-1.1.csv'
    # @%players
    # AUTO_COMPRESS=TRUE
    # OVERWRITE=TRUE;
    # """
    cur = connection.cursor()
    cur.execute(put_command)
    cur.close()

    return "success"