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