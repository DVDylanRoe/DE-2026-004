import snowflake.connector


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


def stage_data(connection, put_command):
    cur = connection.cursor()
    cur.execute(put_command)
    cur.close()

    return "success"


def copy_data_into_table(connection, copy_command):
    cur = connection.cursor()

    cur.execute(copy_command)
    cur.close()
