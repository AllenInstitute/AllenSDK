import logging
from typing import List, Optional
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
    """A data fetching class that serves as an API for fetching 'raw'
    data from LIMS necessary (but not sufficient) for filling
    a 'BehaviorOphysSession'.

    Most 'raw' data provided by this API needs to be processed by
    BehaviorOphysDataXforms methods in order to usable by
    'BehaviorOphysSession's.
    """

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

    def get_ophys_experiment_id(self) -> int:
        return self.ophys_experiment_id

    @memoize
    def get_ophys_session_id(self) -> int:
        """Get the ophys session id associated with the ophys experiment
        id used to initialize the API"""
        query = """
                SELECT os.id FROM ophys_sessions os
                JOIN ophys_experiment oe ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_experiment_container_id(self) -> int:
        """Get the experiment container id associated with the ophys
        experiment id used to initialize the API"""
        query = """
                SELECT visual_behavior_experiment_container_id
                FROM ophys_experiments_visual_behavior_experiment_containers
                WHERE ophys_experiment_id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=False)

    @memoize
    def get_behavior_stimulus_file(self) -> str:
        """Get the filepath to the StimulusPickle file for the session
        associated with the ophys experiment id used to initialize the API"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS stim_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs ON bs.ophys_session_id = os.id
                LEFT JOIN well_known_files wkf ON wkf.attachable_id = bs.id
                JOIN well_known_file_types wkft
                ON wkf.well_known_file_type_id = wkft.id
                WHERE wkf.attachable_type = 'BehaviorSession'
                AND wkft.name = 'StimulusPickle'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_nwb_filepath(self) -> str:
        """Get the filepath of the nwb file associated with the ophys
        experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename AS nwb_file
                FROM ophys_experiments oe
                JOIN well_known_files wkf ON wkf.attachable_id = oe.id
                JOIN well_known_file_types wkft
                ON wkf.well_known_file_type_id = wkft.id
                WHERE wkft.name ='BehaviorOphysNwb'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_eye_tracking_filepath(self) -> str:
        """Get the filepath of the eye tracking file (*.h5) associated with the
        ophys experiment"""
        query = """
                SELECT wkf.storage_directory || wkf.filename
                AS eye_tracking_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf
                ON wkf.attachable_id = oe.ophys_session_id
                JOIN well_known_file_types wkft
                ON wkf.well_known_file_type_id = wkft.id
                WHERE wkf.attachable_type = 'OphysSession'
                AND wkft.name = 'EyeTracking Ellipses'
                AND oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @staticmethod
    def get_ophys_experiment_df() -> pd.DataFrame:
        """Get a DataFrame of metadata for ophys experiments"""
        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
               (PostgresQueryMixin)())
        query = """
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
            LEFT JOIN specimens sp ON sp.id = os.specimen_id
            LEFT JOIN donors d ON d.id = sp.donor_id
            LEFT JOIN imaging_depths id ON id.id = oe.imaging_depth_id
            LEFT JOIN structures st ON st.id = oe.targeted_structure_id
            LEFT JOIN equipment ON equipment.id = os.equipment_id;
            """

        return pd.read_sql(query, api.get_connection())

    @staticmethod
    def get_containers_df(only_passed=True) -> pd.DataFrame:
        """Get a DataFrame of experiment containers"""

        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
               (PostgresQueryMixin)())
        if only_passed is True:
            query = """
                    SELECT *
                    FROM visual_behavior_experiment_containers vbc
                    WHERE workflow_state IN ('container_qc','publish');
                    """
        else:
            query = """
                    SELECT *
                    FROM visual_behavior_experiment_containers vbc;
                    """

        return pd.read_sql(query, api.get_connection()).rename(
            columns={'id': 'container_id'})[['container_id',
                                             'specimen_id',
                                             'workflow_state']]

    @classmethod
    def get_api_list_by_container_id(cls, container_id
                                     ) -> List["BehaviorOphysLimsApi"]:
        """Return a list of BehaviorOphysLimsApi instances for all
        ophys experiments"""
        df = cls.get_ophys_experiment_df()
        container_selector = df['container_id'] == container_id
        oeid_list = df[container_selector]['ophys_experiment_id'].values
        return [cls(oeid) for oeid in oeid_list]


if __name__ == "__main__":

    print(BehaviorOphysLimsApi.get_ophys_experiment_df())
