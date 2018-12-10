import os
import six

import pandas as pd
import psycopg2
import psycopg2.extras

DEFAULT_DATABASE = 'lims2'
DEFAULT_HOST = 'limsdb2'
DEFAULT_PORT = 5432
DEFAULT_USERNAME = 'limsreader'


def psycopg2_select(query, database=DEFAULT_DATABASE, host=DEFAULT_HOST, port=DEFAULT_PORT, username=DEFAULT_USERNAME, password=None):

    if password is None:
        password = os.environ.get('LIMS2_PASSWORD', None)

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


def clean_multiline_query(query):
    return ' '.join([line.strip() for line in query.splitlines()])


def produce_in_clause_target(values, sep=',', left='(', right=')'):
    return left + sep.join([
        str(val) if not isinstance(val, six.string_types) else '\'{}\''.format(val)
        for val in values
        ]) + right


class LimsApi(object):

    def __init__(self, query_fn=None, *args, **kwargs):

        if query_fn is None:
            query_fn = psycopg2_select
        self.query_fn = query_fn

        super(LimsApi, self).__init__(*args, **kwargs)


    def get_well_known_file_table(self, attachable_types=None, attachable_ids=None, file_type_names=None):
        '''Grab (a subset of) the well_known_files table from LIMS.
        '''

        query = clean_multiline_query('''
        select wkf.*, wkft.name as file_type_name from well_known_files wkf
        join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
        ''')

        where = []
        if attachable_types is not None:
            where.append('wkf.attachable_type in {}'.format(produce_in_clause_target(attachable_types)))
        if attachable_ids is not None:
            where.append('wkf.attachable_id in {}'.format(produce_in_clause_target(attachable_ids)))
        if file_type_names is  not None:
            where.append('wkft.name in {}'.format(produce_in_clause_target(file_type_names)))

        if len(where) > 0:
            query = '{} where {}'.format(query, ' and '.join(where))

        response = self.query_fn(query)
        response['path'] = response.apply(lambda row: os.path.join(row['storage_directory'], row['filename']), axis=1)
        response = response.drop(columns=['storage_directory', 'filename'])
        return response


    def _get_well_known_file_path(self, attachable_id, well_known_file_type_name, attachable_type):
        ''' A utility function for getting a single well known file for a single attachable (EcephysSession or EcephysProbe). Human users will 
        be better served by looking at get_well_known_file_table

        Notes
        -----
        only valid for LIMS

        '''

        query = clean_multiline_query('''
            select wkf.storage_directory, wkf.filename from well_known_files wkf
            join well_known_file_types wkft on wkf.well_known_file_type_id = wkft.id
            where wkf.attachable_type = '{}'
            and wkf.attachable_id in ({})
            and wkft.name = \'{}\'
        '''.format(attachable_type, attachable_id, well_known_file_type_name))
        response = self.query_fn(query)
        return os.path.join(response.loc[0, 'storage_directory'], response.loc[0, 'filename'])