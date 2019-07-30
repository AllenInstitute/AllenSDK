
import psycopg2
import psycopg2.extras
import pandas as pd
import logging
import json
import os
from allensdk.internal.core.lims_utilities import safe_system_path

from . import PostgresQueryMixin, OneOrMoreResultExpectedError
from allensdk.api.cache import memoize

from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

logger = logging.getLogger(__name__)

class MesoscopeSessionLimsApi(PostgresQueryMixin):

    def __init__(self, session_id):
        self.session_id = session_id
        self.experiment_ids = None
        self.pairs = None
        self.splitting_json = None
        self.session_df = None
        super().__init__()

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
        return session_folder.values[0]

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


class MesoscopePlaneLimsApi(BehaviorOphysLimsApi):

    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        super().__init__(experiment_id)

    def get_experiment_id(self):
        return self.experiment_id

    def get_metadata(self):
        raise NotImplementedError

    def get_experiment_df(self, experiment_id):
        session_df = self.get_session_df()
        experiment_df = session_df.loc[session_df['experiment_id'] == experiment_id]
        return experiment_df










