
from typing import Optional

from ..core import lims_utilities
from . import PostgresQueryMixin
import pandas as pd

from allensdk.api.cache import memoize
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.trials_processing import get_extended_trials
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.core.authentication import DbCredentials, credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP


class BehaviorLimsApi:

    def __init__(self, behavior_experiment_id: int,
                 lims_credentials: Optional[DbCredentials] = None):
        """
        Notes
        -----
        - behavior_experiment_id is the same as behavior_session_id which is in lims
        - behavior_experiment_id is associated with foraging_id in lims
        - foraging_id in lims is the same as behavior_session_uuid in mtrain which is the same
        as session_uuid in the pickle returned by behavior_stimulus_file
        """
        self.behavior_experiment_id = behavior_experiment_id
        if lims_credentials:
            self.lims_db = PostgresQueryMixin(
                dbname=lims_credentials.dbname, user=lims_credentials.user,
                host=lims_credentials.host, password=lims_credentials.password,
                port=lims_credentials.port)
        else:
            # Use default credentials from provider
            # Currying is equivalent to decorator syntactic sugar
            self.lims_db = (
                credential_injector(LIMS_DB_CREDENTIAL_MAP)
                (PostgresQueryMixin)())

    def get_behavior_experiment_id(self):
        return self.behavior_experiment_id

    @memoize
    def get_behavior_stimulus_file(self):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file 
                FROM behavior_sessions bs 
                LEFT JOIN well_known_files stim ON stim.attachable_id=bs.id AND stim.attachable_type = 'BehaviorSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle') 
                WHERE bs.id= {};
                '''.format(self.get_behavior_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    def get_extended_trials(self):
        filename = self.get_behavior_stimulus_file()
        data = pd.read_pickle(filename)
        return get_extended_trials(data)

    @staticmethod
    def foraging_id_to_behavior_session_id(foraging_id):
        '''maps foraging_id to behavior_session_id'''
        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
               (PostgresQueryMixin)())
        query = '''select id from behavior_sessions where foraging_id = '{}';'''.format(
            foraging_id)
        return api.fetchone(query, strict=True)

    @staticmethod
    def behavior_session_id_to_foraging_id(behavior_session_id):
        '''maps behavior_session_id to foraging_id'''
        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
               (PostgresQueryMixin)())
        query = '''select foraging_id from behavior_sessions where id = '{}';'''.format(
            behavior_session_id)
        return api.fetchone(query, strict=True)

    @classmethod
    def from_foraging_id(cls, foraging_id: str,
                         lims_credentials: Optional[DbCredentials] = None):
        return cls(
            behavior_experiment_id=cls.foraging_id_to_behavior_session_id(
                foraging_id, lims_credentials))
