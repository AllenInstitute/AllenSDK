from typing import Optional

import psycopg2
import psycopg2.extras
import pandas as pd

from allensdk import one, OneResultExpectedError
from allensdk.core.authentication import DbCredentials, credential_injector


class OneOrMoreResultExpectedError(RuntimeError):
    pass


def psycopg2_select(query, database, host, port, username, password):

    connection = psycopg2.connect(
        host=host, port=port, dbname=database,
        user=username, password=password,
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
    def __init__(self, *, dbname, user, host, password, port):

        self.dbname = dbname
        self.user = user
        self.host = host
        self.password = password
        self.port = port

    def get_cursor(self):
        return self.get_connection().cursor()

    def get_connection(self):
        return psycopg2.connect(dbname=self.dbname, user=self.user,
                                host=self.host, password=self.password,
                                port=self.port)

    def fetchone(self, query, strict=True):
        response = one(list(self.select(query).to_dict().values()))
        if strict is True and (len(response) != 1 or response[0] is None):
            raise OneResultExpectedError
        return response[0]

    def fetchall(self, query, strict=True):
        response = self.select(query)
        return [one(x) for x in response.values.flat]

    def select(self, query):
        return psycopg2_select(
            query,
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


def db_connection_creator(default_credentials: dict,
                          credentials: Optional[DbCredentials] = None,
                          ) -> PostgresQueryMixin:
    """Create a db connection using either default credentials or
    custom credentials.

    Parameters
    ----------
    default_credentials : dict
        Default credentials to use for creating the DB connection.
        Some options can be found in allensdk.core.auth_config.
    credentials : Optional[DbCredentials], optional
        User specified credentials, by default None

    Returns
    -------
    PostgresQueryMixin
        A DB connection instance which can execute queries to the DB
        specified by credentials or default_credentials.
    """
    if credentials:
        db_conn = PostgresQueryMixin(
            dbname=credentials.dbname, user=credentials.user,
            host=credentials.host, port=credentials.port,
            password=credentials.password)
    else:
        db_conn = (credential_injector(default_credentials)
                   (PostgresQueryMixin)())

    return db_conn
