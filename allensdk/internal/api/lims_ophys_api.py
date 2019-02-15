from . import PostgresQueryMixin

from allensdk.api.cache import memoize

class LimsOphysAPI(PostgresQueryMixin):

    @memoize
    def get_ophys_experiment_dir(self, ophys_experiment_id=None):
        query = '''
                SELECT oe.storage_directory
                FROM ophys_experiments oe
                WHERE oe.id = {ophys_experiment_id};
                '''.format(ophys_experiment_id=ophys_experiment_id)
        return self.fetchone(query, strict=True)

    @memoize
    def get_nwb_filepath(self, ophys_experiment_id=None):

        query = '''
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'NWBOphys')
                WHERE oe.id = {ophys_experiment_id};
                '''.format(ophys_experiment_id=ophys_experiment_id)
        return self.fetchone(query, strict=True)
