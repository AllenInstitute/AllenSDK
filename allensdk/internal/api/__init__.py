import psycopg2
import psycopg2.extras
import pandas as pd


class OneResultExpectedError(RuntimeError):
    pass


class OneOrMoreResultExpectedError(RuntimeError):
    pass


def one(x):
    if isinstance(x, str):
        return x
    if len(x) != 1:
        raise OneResultExpectedError('Expected length one result, received: {} results from query'.format(x))
    if isinstance(x, set):
        return list(x)[0]
    else:
        return x[0]


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


class PostgresQueryMixin(object):

    def __init__(self, dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432):

        self.dbname = dbname
        self.user = user
        self.host = host
        self.password = password
        self.port = port

    def get_cursor(self):
        return self.get_connection().cursor()

    def get_connection(self):
        return psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, password=self.password, port=self.port)

    def fetchone(self, query, strict=True):
        if strict is True:
            response = list(self.select(query).to_dict().values())
            return one(one(response))
        response = list(self.select_one(query).values())
        return one(response)

    def fetchall(self, query, strict=True):
        response = self.select(query)
        return [one(x) for x in response.values.flat]

    def select(self, query):
        return psycopg2_select(query, 
            database=self.dbname, 
            host=self.host, 
            port=self.port, 
            username=self.user, 
            password=self.password
        )

    def select_one(self, query):
        data = self.select(query).to_dict('record')
        if len(data) == 1:
            return data[0]
        return {}
