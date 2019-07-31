
import psycopg2
import psycopg2.extras
import pandas as pd
import logging
import json
import os

from pandas import DataFrame

from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.brain_observatory.behavior.sync import get_sync_data

from . import PostgresQueryMixin


from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

logger = logging.getLogger(__name__)


class MesoscopeSessionLimsApi(PostgresQueryMixin):

    def __init__(self, session_id):
        self.session_id = session_id
        self.experiment_ids = None
        self.pairs = None
        self.splitting_json = None
        self.session_folder = None
        self.session_df = None
        self.sync_path = None
        self.planes_timestamps = None
        super().__init__()

    def get_well_known_file(self, file_type):
        """Gets a well_known_file's location"""
        query = ' '.join(['SELECT wkf.storage_directory, wkf.filename FROM well_known_files wkf',
                           'JOIN well_known_file_types wkft',
                           'ON wkf.well_known_file_type_id = wkft.id',
                           'WHERE',
                           'attachable_id = {}'.format(self.session_id),
                           'AND wkft.name = \'{}\''.format(file_type)])

        query = query.format(self.session_id)
        filepath = pd.read_sql(query, self.get_connection())
        return filepath

    def get_session_id(self):
        return self.session_id

    def get_session_experiments(self):
        query = ' '.join((
            "SELECT oe.id as experiment_id",
            "FROM ophys_experiments oe",
            "WHERE oe.ophys_session_id = {}"
        ))
        self.experiment_ids = pd.read_sql(query.format(self.get_session_id()), self.get_connection())
        return self.experiment_ids

    def get_session_folder(self):
        _session = pd.DataFrame(self.get_session_df())
        session_folder = _session['session_folder']
        self.session_folder = safe_system_path(session_folder.values[0])
        return self.session_folder

    def get_session_df(self):
        query = ' '.join(("SELECT oe.id as experiment_id, os.id as session_id",
                    ", os.storage_directory as session_folder, oe.storage_directory as experiment_folder",
                    ", sp.name as specimen",
                    ", os.date_of_acquisition as date",
                    ", imaging_depths.depth as depth",
                    ", st.acronym as structure",
                    ", os.parent_session_id as parent_id",
                    ", oe.workflow_state",
                    ", os.stimulus_name as stimulus",
                    " FROM ophys_experiments oe",
                    "join ophys_sessions os on os.id = oe.ophys_session_id "
                    "join specimens sp on sp.id = os.specimen_id "
                    "join projects p on p.id = os.project_id "
                    "join imaging_depths on imaging_depths.id = oe.imaging_depth_id "
                    "join structures st on st.id = oe.targeted_structure_id "
                    " WHERE p.code in ('MesoscopeDevelopment', 'VisualBehaviorMultiscope') "
                    " AND oe.workflow_state in ('processing', 'qc', 'passed', 'failed') "
                    " AND os.workflow_state ='uploaded' "
                    " AND os.id='{}' ",
        ))
        query = query.format(self.get_session_id())
        self.session_df = pd.read_sql(query, self.get_connection())
        return self.session_df

    def get_splitting_json(self):
        session_folder = self.get_session_folder()
        """this info should not be read form splitting json, but right now this info is not stored in the database"""
        json_path = os.path.join(session_folder, f"MESOSCOPE_FILE_SPLITTING_QUEUE_{self.session_id}_input.json")
        self.splitting_json = safe_system_path(json_path)
        if not os.path.isfile(self.splitting_json):
            logger.error("Unable to find splitting json for session: {}".format(self.session_id))
        return self.splitting_json

    def get_paired_experiments(self):
        splitting_json = self.get_splitting_json()
        self.pairs = []
        with open(splitting_json, "r") as f:
            data = json.load(f)
        for pg in data.get("plane_groups", []):
            self.pairs.append([p["experiment_id"] for p in pg.get("ophys_experiments", [])])
        return self.pairs

    def get_sync_file(self):
            sync_file_df = self.get_well_known_file(file_type='OphysRigSync')
            sync_file_dir = safe_system_path(sync_file_df['storage_directory'].values[0])
            sync_file_name = sync_file_df['filename'].values[0]
            return os.path.join(sync_file_dir, sync_file_name)

    def get_sync_data(self):
        sync_path = self.get_sync_file()
        return get_sync_data(sync_path)

    def split_session_timestamps(self):

        #this needs a check for dropped frames: compare timestamps with scanimage header's timestamps.

        timestamps = self.get_sync_data()['ophys_frames']
        planes_timestamps = pd.DataFrame(columns= ['plane_id', 'ophys_timestamp'], index = range(len(self.get_session_experiments())))
        pairs = self.get_paired_experiments()
        i = 0
        for pair in range(len(pairs)):
            planes_timestamps['plane_id'][i] = pairs[pair][0]
            planes_timestamps['plane_id'][i+1] = pairs[pair][1]
            planes_timestamps['ophys_timestamp'][i] = planes_timestamps['ophys_timestamp'][i+1] = timestamps[pair::len(pairs)]
            i += 2
        self.planes_timestamps = planes_timestamps
        return self.planes_timestamps


class MesoscopePlaneLimsApi(BehaviorOphysLimsApi):

    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.session_id = None
        self.experiment_df = None
        super().__init__(experiment_id)

    def get_ophys_timestamps(self):
        session = MesoscopeSessionLimsApi(self.session_id)
        session_timestamps = session.split_session_timestamps()
        plane_timestamps = session_timestamps.loc[session_timestamps['plane_id'] == self.ophys_experiment_id]
        self.ophys_timestamps = plane_timestamps
        return self.ophys_timestamps

    def get_experiment_df(self):

        api = PostgresQueryMixin()
        query = ''' 
                SELECT 
                
                oe.id as experiment_id, 
                os.id as session_id, 
                oe.storage_directory as experiment_folder,
                sp.name as specimen,
                os.date_of_acquisition as date,
                imaging_depths.depth as depth,
                st.acronym as structure,
                os.parent_session_id as parent_id,
                oe.workflow_state as workflow_state,
                os.stimulus_name as stimulus
                
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON os.id = oe.ophys_session_id 
                JOIN specimens sp ON sp.id = os.specimen_id  
                JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id 
                JOIN structures st ON st.id = oe.targeted_structure_id 
                
                AND oe.id='{}'
                '''

        query = query.format(self.get_ophys_experiment_id())
        self.experiment_df = pd.read_sql(query, api.get_connection())
        return self.experiment_df


    def get_ophys_session_id(self):
        self.get_experiment_df()
        self.session_id = self.experiment_df['session_id'].values[0]
        return self.session_id

    # def get_metadata(self):
    #     raise NotImplementedError













