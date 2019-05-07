import psycopg2
import psycopg2.extras
import pandas as pd


LIMS_DB_NAME = 'lims2'
LIMS_HOST = 'limsdb2'
LIMS_PORT = 5432
READ_ONLY_USERNAME = 'limsreader'
READ_ONLY_PASSWORD = 'limsro'


def psycopg2_select(query, database, host, port, username, password):

    connection = psycopg2.connect(
        host=host, port=port, dbname=database, user=username, password=password,
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        response = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    return pd.DataFrame(response)


class LimsApiMixin:

    def __init__(self, 
        database=LIMS_DB_NAME, 
        host=LIMS_HOST, 
        port=LIMS_PORT, 
        username=READ_ONLY_USERNAME, 
        password=READ_ONLY_PASSWORD
    ):
        self.database = database
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def select(self, query):
        return psycopg2_select(query, 
            database=self.database, 
            host=self.host, 
            port=self.port, 
            username=self.username, 
            password=self.password
        )

    def select_one(self, query):
        data = self.select(query)
        if self.data.shape[0] != 1:
            raise ValueError(f'expected exactly 1 result, found {repr(data)}')
        return data.to_dict('record')[0]