import logging
from typing import List, Optional

import pandas as pd
from allensdk.api.warehouse_cache.cache import memoize
from allensdk.brain_observatory.behavior.session_apis.abcs. \
    data_extractor_base.behavior_ophys_data_extractor_base import \
    BehaviorOphysDataExtractorBase
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsExtractor, OphysLimsExtractor)
from allensdk.brain_observatory.behavior.session_apis.data_transforms import \
    BehaviorOphysDataTransforms
from allensdk.core.auth_config import (LIMS_DB_CREDENTIAL_MAP,
                                       MTRAIN_DB_CREDENTIAL_MAP)
from allensdk.core.authentication import DbCredentials
from allensdk.core.cache_method_utilities import CachedInstanceMethodMixin
from allensdk.internal.api import db_connection_creator
from allensdk.internal.core.lims_utilities import safe_system_path


class BehaviorOphysLimsApi(BehaviorOphysDataTransforms,
                           CachedInstanceMethodMixin):
    """A data fetching and processing class that serves processed data from
    a specified data source (extractor). Contains all methods
    needed to populate a BehaviorOphysExperiment."""

    def __init__(self,
                 ophys_experiment_id: Optional[int] = None,
                 lims_credentials: Optional[DbCredentials] = None,
                 mtrain_credentials: Optional[DbCredentials] = None,
                 extractor: Optional[BehaviorOphysDataExtractorBase] = None,
                 skip_eye_tracking: bool = False):

        if extractor is None:
            if ophys_experiment_id is not None:
                extractor = BehaviorOphysLimsExtractor(
                    ophys_experiment_id,
                    lims_credentials,
                    mtrain_credentials)
            else:
                raise RuntimeError(
                    "BehaviorOphysLimsApi must be provided either an "
                    "instantiated 'extractor' or an 'ophys_experiment_id'!")

        super().__init__(extractor=extractor,
                         skip_eye_tracking=skip_eye_tracking)


class BehaviorOphysLimsExtractor(OphysLimsExtractor, BehaviorLimsExtractor,
                                 BehaviorOphysDataExtractorBase):
    """A data fetching class that serves as an API for fetching 'raw'
    data from LIMS necessary (but not sufficient) for filling
    a 'BehaviorOphysExperiment'.

    Most 'raw' data provided by this API needs to be processed by
    BehaviorOphysDataTransforms methods in order to usable by
    'BehaviorOphysExperiment's.
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
                JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                """.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_project_code(self) -> str:
        """Get the project code"""
        query = f"""
            SELECT projects.code AS project_code
            FROM ophys_sessions
            JOIN projects ON projects.id = ophys_sessions.project_id
            WHERE ophys_sessions.id = {self.get_ophys_session_id()}
        """
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_ophys_container_id(self) -> int:
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
        query = f"""
                SELECT wkf.storage_directory || wkf.filename AS eye_tracking_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files wkf ON wkf.attachable_id = oe.ophys_session_id
                JOIN well_known_file_types wkft ON wkf.well_known_file_type_id = wkft.id
                WHERE wkf.attachable_type = 'OphysSession'
                    AND wkft.name = 'EyeTracking Ellipses'
                    AND oe.id = {self.get_ophys_experiment_id()};
                """  # noqa E501
        return safe_system_path(self.lims_db.fetchone(query, strict=True))

    @memoize
    def get_eye_tracking_rig_geometry(self) -> Optional[dict]:
        """Get the eye tracking rig geometry metadata"""
        ophys_experiment_id = self.get_ophys_experiment_id()

        query = f'''
            SELECT oec.*, oect.name as config_type, equipment.name as equipment_name
            FROM ophys_sessions os
            JOIN observatory_experiment_configs oec ON oec.equipment_id = os.equipment_id
            JOIN observatory_experiment_config_types oect ON oect.id = oec.observatory_experiment_config_type_id
            JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
            JOIN equipment ON equipment.id = oec.equipment_id
            WHERE oe.id = {ophys_experiment_id} AND 
                oec.active_date <= os.date_of_acquisition AND
                oect.name IN ('eye camera position', 'led position', 'screen position')
        '''  # noqa E501
        # Get the raw data
        rig_geometry = pd.read_sql(query, self.lims_db.get_connection())

        if rig_geometry.empty:
            # There is no rig geometry for this experiment
            return None

        return self._process_eye_tracking_rig_geometry(
            rig_geometry=rig_geometry
        )

    def get_ophys_experiment_df(self) -> pd.DataFrame:
        """Get a DataFrame of metadata for ophys experiments"""
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

        return pd.read_sql(query, self.lims_db.get_connection())

    def get_containers_df(self, only_passed=True) -> pd.DataFrame:
        """Get a DataFrame of experiment containers"""
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

        return pd.read_sql(query, self.lims_db.get_connection()).rename(
            columns={'id': 'container_id'})[['container_id',
                                             'specimen_id',
                                             'workflow_state']]

    @staticmethod
    def _process_eye_tracking_rig_geometry(rig_geometry: pd.DataFrame) -> dict:
        """
        Processes the raw eye tracking rig geometry returned by LIMS
        """
        # Map the config types to new names
        rig_geometry_config_type_map = {
            'eye camera position': 'camera',
            'screen position': 'monitor',
            'led position': 'led'
        }
        rig_geometry['config_type'] = rig_geometry['config_type'] \
            .map(rig_geometry_config_type_map)

        # Select the most recent config
        # that precedes the date_of_acquisition for this experiment
        rig_geometry = rig_geometry.sort_values('active_date', ascending=False)
        rig_geometry = rig_geometry.groupby('config_type') \
            .apply(lambda x: x.iloc[0])

        # Construct dictionary for positions
        position = rig_geometry[['center_x_mm', 'center_y_mm', 'center_z_mm']]
        position.index = [
            f'{v}_position_mm' if v != 'led'
            else f'{v}_position' for v in position.index]
        position = position.to_dict(orient='index')
        position = {
            config_type: [
                values['center_x_mm'],
                values['center_y_mm'],
                values['center_z_mm']
            ]
            for config_type, values in position.items()
        }

        # Construct dictionary for rotations
        rotation = rig_geometry[['rotation_x_deg', 'rotation_y_deg',
                                 'rotation_z_deg']]
        rotation = rotation[rotation.index != 'led']
        rotation.index = [f'{v}_rotation_deg' for v in rotation.index]
        rotation = rotation.to_dict(orient='index')
        rotation = {
            config_type: [
                values['rotation_x_deg'],
                values['rotation_y_deg'],
                values['rotation_z_deg']
            ] for config_type, values in rotation.items()
        }

        # Combine the dictionaries
        return {
            **position,
            **rotation,
            'equipment': rig_geometry['equipment_name'].iloc[0]
        }

    @classmethod
    def get_api_list_by_container_id(cls, container_id
                                     ) -> List["BehaviorOphysLimsApi"]:
        """Return a list of BehaviorOphysLimsApi instances for all
        ophys experiments"""
        df = cls.get_ophys_experiment_df()
        container_selector = df['container_id'] == container_id
        oeid_list = df[container_selector]['ophys_experiment_id'].values
        return [cls(oeid) for oeid in oeid_list]

    @memoize
    def get_event_detection_filepath(self) -> str:
        """Gets the filepath to the event detection data"""
        query = f'''
            SELECT wkf.storage_directory || wkf.filename AS event_detection_filepath
            FROM ophys_experiments oe
            LEFT JOIN well_known_files wkf ON wkf.attachable_id = oe.id
            JOIN well_known_file_types wkft ON wkf.well_known_file_type_id = wkft.id
            WHERE wkft.name = 'OphysEventTraceFile'
                AND oe.id = {self.get_ophys_experiment_id()};
        '''  # noqa E501
        return safe_system_path(self.lims_db.fetchone(query, strict=True))


if __name__ == "__main__":
    print(BehaviorOphysLimsApi.get_ophys_experiment_df())
