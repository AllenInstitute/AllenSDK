
from ..core import lims_utilities
from . import PostgresQueryMixin
import pandas as pd


class BehaviorLimsApi(PostgresQueryMixin):

    def foraging_id_to_behavior_session_id(self, foraging_id):
        '''maps foraging_id to behavior_session_id'''
        query = '''select id from behavior_sessions where foraging_id = '{}';'''.format(
            foraging_id)
        return self.fetchone(query, strict=True)

    def behavior_session_id_to_foraging_id(self, behavior_session_id):
        '''maps behavior_session_id to foraging_id'''
        query = '''select foraging_id from behavior_sessions where id = '{}';'''.format(
            behavior_session_id)
        return self.fetchone(query, strict=True)
