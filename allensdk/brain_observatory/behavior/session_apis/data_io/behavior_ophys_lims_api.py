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
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.core.authentication import credential_injector, DbCredentials
from allensdk.brain_observatory.behavior.session_apis.data_transforms import (
    BehaviorOphysDataXforms)


class BehaviorOphysLimsApi(BehaviorLimsApi, OphysLimsApi,
                           BehaviorOphysDataXforms):

    def __init__(self, ophys_experiment_id: int,
                 lims_credentials: Optional[DbCredentials] = None,
                 mtrain_credentials: Optional[DbCredentials] = None):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.lims_db = db_connection_creator(
            credentials=lims_credentials,
            default_credentials=LIMS_DB_CREDENTIAL_MAP)

        self.ophys_experiment_id = ophys_experiment_id
        super().__init__(behavior_session_id=self.get_behavior_session_id(),
                         lims_credentials=lims_credentials,
                         mtrain_credentials=mtrain_credentials)

    @memoize
    def get_ophys_timestamps(self):

        ophys_timestamps = self.get_sync_data()['ophys_frames']
        dff_traces = self.get_raw_dff_data()
        plane_group = self.get_imaging_plane_group()

        number_of_cells, number_of_dff_frames = dff_traces.shape
        # Scientifica data has extra frames in the sync file relative
        # to the number of frames in the video. These sentinel frames
        # should be removed.
        # NOTE: This fix does not apply to mesoscope data.
        # See http://confluence.corp.alleninstitute.org/x/9DVnAg
        if plane_group is None:    # non-mesoscope
            num_of_timestamps = len(ophys_timestamps)
            if (number_of_dff_frames < num_of_timestamps):
                self.logger.info(
                    "Truncating acquisition frames ('ophys_frames') "
                    f"(len={num_of_timestamps}) to the number of frames "
                    f"in the df/f trace ({number_of_dff_frames}).")
                ophys_timestamps = ophys_timestamps[:number_of_dff_frames]
            elif number_of_dff_frames > num_of_timestamps:
                raise RuntimeError(
                    f"dff_frames (len={number_of_dff_frames}) is longer "
                    f"than timestamps (len={num_of_timestamps}).")
        # Mesoscope data
        # Resample if collecting multiple concurrent planes (e.g. mesoscope)
        # because the frames are interleaved
        else:
            group_count = self.get_plane_group_count()
            self.logger.info(
                "Mesoscope data detected. Splitting timestamps "
                f"(len={len(ophys_timestamps)} over {group_count} "
                "plane group(s).")
            ophys_timestamps = self._process_ophys_plane_timestamps(
                ophys_timestamps, plane_group, group_count)
            num_of_timestamps = len(ophys_timestamps)
            if number_of_dff_frames != num_of_timestamps:
                raise RuntimeError(
                    f"dff_frames (len={number_of_dff_frames}) is not equal to "
                    f"number of split timestamps (len={num_of_timestamps}).")
        return ophys_timestamps

    def get_behavior_session_id(self):
        query = '''
                SELECT bs.id FROM behavior_sessions bs
                JOIN ophys_sessions os ON os.id = bs.ophys_session_id
                JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=True)

    @memoize
    def get_ophys_session_id(self):
        query = '''
                SELECT os.id FROM ophys_sessions os
                JOIN ophys_experiment oe ON oe.ophys_session_id = os.id
                WHERE oe.id = {};
                '''.format(self.get_ophys_experiment_id())
        return self.lims_db.fetchone(query, strict=False)

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
