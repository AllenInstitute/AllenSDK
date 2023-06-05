CREDENTIAL_KEYS = [
    # key, default value
    ("LIMS_DBNAME", None),
    ("LIMS_USER", None),
    ("LIMS_HOST", None),
    ("LIMS_PORT", 5432),
    ("LIMS_PASSWORD", None),
    ("MTRAIN_DBNAME", None),
    ("MTRAIN_USER", None),
    ("MTRAIN_HOST", None),
    ("MTRAIN_PORT", 5432),
    ("MTRAIN_PASSWORD", None)
]

# For PostgresQueryMixin
LIMS_DB_CREDENTIAL_MAP = {  
    "dbname": "lims2",
    "user": "limsreader",
    "host": "limsdb2",
    "password": "limsro",
    "port": 5432
}

# For PostgresQueryMixin
MTRAIN_DB_CREDENTIAL_MAP = {
    "dbname": "MTRAIN_DBNAME",
    "user": "MTRAIN_USER",
    "host": "MTRAIN_HOST",
    "password": "MTRAIN_PASSWORD",
    "port": "MTRAIN_PORT"
}