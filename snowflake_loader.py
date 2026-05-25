import snowflake.connector


def get_connection(credentials):
    conn = snowflake.connector.connect(
        account=credentials.get("account"),
        user=credentials.get("user"),
        authenticator=credentials.get("authenticator"),
        role=credentials.get("role"),
        warehouse=credentials.get("warehouse"),
        database=credentials.get("database"),
        schema=credentials.get("schema"),
    )

    return conn


def stage_data(connection, put_command):
    cur = connection.cursor()
    cur.execute(put_command)
    cur.close()

    return "success"


def copy_data_into_table(connection, copy_command):
    cur = connection.cursor()

    cur.execute(copy_command)
    cur.close()
