import logging
from typing import Optional
import pandas as pd

from allensdk.api.cache import memoize
from allensdk.brain_observatory.behavior.session_apis.data_io.ophys_lims_api \
    import OphysLimsApi
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsApi)
from allensdk.internal.api import db_connection_creator, PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.core.auth_config import (
    LIMS_DB_CREDENTIAL_MAP, MTRAIN_DB_CREDENTIAL_MAP)
from allensdk.core.authentication import credential_injector, DbCredentials
from allensdk.brain_observatory.behavior.session_apis.data_transforms import (
    BehaviorOphysDataXforms)


class BehaviorOphysLimsApi(BehaviorOphysDataXforms,  OphysLimsApi,
                           BehaviorLimsApi):

    def __init__(self, ophys_experiment_id: int,
                 lims_credentials: Optional[DbCredentials] = None,
                 mtrain_credentials: Optional[DbCredentials] = None):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.lims_db = db_connection_creator(
            credentials=lims_credentials,
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

        self.mtrain_db = db_connection_creator(
            credentials=mtrain_credentials,
            fallback_credentials=MTRAIN_DB_CREDENTIAL_MAP)

        self.ophys_experiment_id = ophys_experiment_id
        self.behavior_session_id = self.get_behavior_session_id()

    def get_ophys_experiment_id(self):
        return self.ophys_experiment_id

    @memoize
    def get_ophys_session_id(self):
        query = '''
                SELECT os.id FROM ophys_sessions os
                JOIN ophys_experiment oe ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_experiment_container_id(self):
        query = '''
                SELECT visual_behavior_experiment_container_id
                FROM ophys_experiments_visual_behavior_experiment_containers
                WHERE ophys_experiment_id= {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=False)

    @memoize
    def get_behavior_stimulus_file(self):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs ON bs.ophys_session_id=os.id
                LEFT JOIN well_known_files stim ON stim.attachable_id=bs.id AND stim.attachable_type = 'BehaviorSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
                WHERE oe.id= {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_nwb_filepath(self):

        query = '''
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'BehaviorOphysNwb')
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_eye_tracking_filepath(self):
        query = '''SELECT wkf.storage_directory || wkf.filename AS eye_tracking_file
                   FROM ophys_experiments oe
                   LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.ophys_session_id
                   AND wkf.attachable_type = 'OphysSession'
                   AND wkf.well_known_file_type_id=(SELECT id FROM well_known_file_types WHERE name = 'EyeTracking Ellipses')
                   WHERE oe.id={};
                   '''.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @staticmethod
    def get_ophys_experiment_df():

        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
               (PostgresQueryMixin)())
        query = '''
                SELECT

                oec.visual_behavior_experiment_container_id as container_id,
                oec.ophys_experiment_id,
                oe.workflow_state,
                d.full_genotype as full_genotype,
                id.depth as imaging_depth,
                st.acronym as targeted_structure,
                os.name as session_name,
                equipment.name as equipment_name

                FROM ophys_experiments_visual_behavior_experiment_containers oec
                LEFT JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
                LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN specimens sp ON sp.id=os.specimen_id
                LEFT JOIN donors d ON d.id=sp.donor_id
                LEFT JOIN imaging_depths id ON id.id=oe.imaging_depth_id
                LEFT JOIN structures st ON st.id=oe.targeted_structure_id
                LEFT JOIN equipment ON equipment.id=os.equipment_id
                '''

        return pd.read_sql(query, api.get_connection())

    @staticmethod
    def get_containers_df(only_passed=True):

        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
               (PostgresQueryMixin)())
        if only_passed is True:
            query = '''
                    SELECT *
                    FROM visual_behavior_experiment_containers vbc
                    WHERE workflow_state IN ('container_qc','publish');
                    '''
        else:
            query = '''
                    SELECT *
                    FROM visual_behavior_experiment_containers vbc
                    '''

        return pd.read_sql(query, api.get_connection()).rename(columns={'id': 'container_id'})[['container_id', 'specimen_id', 'workflow_state']]

    @classmethod
    def get_api_list_by_container_id(cls, container_id):

        df = cls.get_ophys_experiment_df()
        oeid_list = df[df['container_id'] == container_id]['ophys_experiment_id'].values
        return [cls(oeid) for oeid in oeid_list]


if __name__ == "__main__":

    print(BehaviorOphysLimsApi.get_ophys_experiment_df())
    # print(BehaviorOphysLimsApi.get_containers_df(only_passed=False))

    # print(BehaviorOphysLimsApi.get_api_by_container(838105949))

    # ophys_experiment_id = df['ophys_experiment_id'].iloc[0]
    # print(ophys_experiment_id)
    # BehaviorOphysLimsApi
    # print(L)
    # for c in sorted(L.columns):
    #     print(c)
    # for x in [791352433, 814796698, 814796612, 814796558, 814797528]:
    #     print(x in L)
