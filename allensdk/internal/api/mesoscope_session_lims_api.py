import pandas as pd
import json
import os
from allensdk.internal.core.lims_utilities import safe_system_path
from . import PostgresQueryMixin
from allensdk.brain_observatory.mesoscope.sync import get_sync_data
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


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
        query = ' '.join(("SELECT oe.id as experiment_id, os.id as session_id, os.name as ses_name"
                    ", os.storage_directory as session_folder, oe.storage_directory as experiment_folder",
                    ", sp.name as specimen",
                    ", os.date_of_acquisition as date",
                    ", oe.calculated_depth as depth",
                    ", st.acronym as structure",
                    ", os.parent_session_id as parent_id",
                    ", oe.workflow_state as wfl_state ",
                    ", users.login as operator "
                    "FROM ophys_experiments oe",
                    "join ophys_sessions os on os.id = oe.ophys_session_id "
                    "join specimens sp on sp.id = os.specimen_id "
                    "join projects p on p.id = os.project_id "
                    "join structures st on st.id = oe.targeted_structure_id "
                    "join users on users.id = os.operator_id"
                    " WHERE os.id='{}' ",
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


    def split_session_timestamps(self):

        #this needs a check for dropped frames: compare timestamps with scanimage header's timestamps.

        timestamps = get_sync_data(self)['ophys_frames']
        planes_timestamps = pd.DataFrame(columns= ['plane_id', 'ophys_timestamps'], index = range(len(self.get_session_experiments())))
        pairs = self.get_paired_experiments()
        i = 0
        for pair in range(len(pairs)):
            planes_timestamps['plane_id'][i] = pairs[pair][0]
            planes_timestamps['plane_id'][i+1] = pairs[pair][1]
            planes_timestamps['ophys_timestamps'][i] = planes_timestamps['ophys_timestamps'][i+1] = timestamps[pair::len(pairs)]
            i += 2
        self.planes_timestamps = planes_timestamps
        return self.planes_timestamps




