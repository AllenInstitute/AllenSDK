import psycopg2

def one(x):
    if len(x) != 1:
        raise RuntimeError('Expected length one input: {}'.format(x))
    if isinstance(x,set):
        return list(x)[0]
    else:
        return x[0]

class LimsQueryMixin(object):
    
    def __init__(self, dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432):

        self.dbname = dbname
        self.user = user
        self.host = host
        self.password = password
        self.port = port

    def fetchone(self, query):
        conn = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, password=self.password, port=self.port)
        cur = conn.cursor()
        cur.execute(query)
        result =  cur.fetchone()
        return one(result) if result is not None else None

