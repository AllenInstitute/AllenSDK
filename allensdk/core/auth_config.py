CREDENTIAL_KEYS = [
    # key, default value
    ("LIMS_DBNAME", "lims2"),
    ("LIMS_USER", "limsreader"),
    ("LIMS_HOST", "limsdb2"),
    ("LIMS_PORT", 5432),
    ("LIMS_PASSWORD", "limsro"),
    ("MTRAIN_DBNAME", None),
    ("MTRAIN_USER", None),
    ("MTRAIN_HOST", None),
    ("MTRAIN_PORT", 5432),
    ("MTRAIN_PASSWORD", None)
]

# For PostgresQueryMixin
LIMS_DB_CREDENTIAL_MAP = {
    "dbname": "LIMS_DBNAME",
    "user": "LIMS_USER",
    "host": "LIMS_HOST",
    "password": "LIMS_PASSWORD",
    "port": "LIMS_PORT"
}

# For PostgresQueryMixin
MTRAIN_DB_CREDENTIAL_MAP = {
    "dbname": "MTRAIN_DBNAME",
    "user": "MTRAIN_USER",
    "host": "MTRAIN_HOST",
    "password": "MTRAIN_PASSWORD",
    "port": "MTRAIN_PORT"
}