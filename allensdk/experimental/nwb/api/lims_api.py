import pandas as pd
import psycopg2
import psycopg2.extras

DEFAULT_DATABASE = 'lims2'
DEFAULT_HOST = 'limsdb2'
DEFAULT_PORT = 5432
DEFAULT_USERNAME = 'limsreader'


def psycopg2_select(query, database=DEFAULT_DATABASE, host=DEFAULT_HOST, port=DEFAULT_PORT, username=DEFAULT_USERNAME):

    connection = psycopg2.connect(
        'host={} port={} dbname={} user={}'.format(host, port, database, username), 
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
    return left + sep.join([str(val) for val in values]) + right



class LimsApi(object):

    def __init__(self, query_fn=None, *args, **kwargs):

        if query_fn is None:
            query_fn = psycopg2_select
        self.query_fn = query_fn

        super(LimsApi, self).__init__(*args, **kwargs)


    def get_well_known_file_table(self):
        '''Queries for a global table of all well known files and their paths, for all experiments and probes

        Notes
        -----
        only valid for LIMS

        '''

        session_query = clean_multiline_query('''
            select wkft.name as file_type, wkf.storage_directory, wkf.filename, wkf.attachable_id as session_id
            from well_known_files wkf
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where wkf.attachable_type = \'EcephysSession\'
        ''')
        session_response = self.query_fn(session_query)
        session_response['probe_id'] = None

        probe_query = clean_multiline_query('''
            select wkft.name as file_type, wkf.storage_directory, wkf.filename, wkf.attachable_id as probe_id, ep.ecephys_session_id as session_id
            from well_known_files wkf
            join ecephys_probes ep on ep.id = wkf.attachable_id
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where wkf.attachable_type = \'EcephysProbe\'
        ''')
        probe_response = self.query_fn(probe_query)
        
        output = pd.concat([session_response, probe_response], sort=False)
        output['path'] = output.apply(lambda row: os.path.join(row['storage_directory'], row['filename']), axis=1)
        output = output.drop(columns=['storage_directory', 'filename'])
        return output.sort_values(by='session_id')


    def _get_well_known_file_path(self, attachable_id, well_known_file_type_name, attachable_type):
        ''' A utility function for getting a single well known file for a single attachable (EcephysSession or EcephysProbe). Human users will 
        be better served by looking at get_well_known_file_table

        Notes
        -----
        only valid for LIMS

        '''

        query: str = clean_multiline_query('''
            select wkf.storage_directory, wkf.filename from well_known_files wkf
            join well_known_file_types wkft on wkf.well_known_file_type_id = wkft.id
            where wkf.attachable_type = '{}'
            and wkf.attachable_id in ({})
            and wkft.name = \'{}\'
        '''.format(attachable_type, attachable_id, well_known_file_type_name))
        response = self.query_fn(query)
        return os.path.join(response.loc[0, 'storage_directory'], response.loc[0, 'filename'])