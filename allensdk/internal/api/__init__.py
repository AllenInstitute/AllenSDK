import psycopg2


class OneResultExpectedError(RuntimeError):
    pass


class OneOrMoreResultExpectedError(RuntimeError):
    pass


def one(x):
    if len(x) != 1:
        raise OneResultExpectedError('Expected length one result, received: {} results form query'.format(x))
    if isinstance(x, set):
        return list(x)[0]
    else:
        return x[0]


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
        cur = self.get_cursor()
        cur.execute(query)
        if strict is True:
            return one(one(cur.fetchall()))
        else:
            result = cur.fetchone()
            return one(result) if result is not None else None

    def fetchall(self, query, strict=True):
        cur = self.get_cursor()
        cur.execute(query)
        return [one(x) for x in cur.fetchall()]
