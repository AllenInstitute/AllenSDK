import numpy as np
import pandas as pd
import uuid

from allensdk.api.cache import memoize
from allensdk.brain_observatory.behavior.behavior_ophys_api import BehaviorOphysApiBase
from allensdk.brain_observatory.behavior.trials_processing import get_extended_trials
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

class BehaviorOnlyLimsApi(BehaviorOphysLimsApi):

    def __init__(self, behavior_experiment_id):
        '''
        For loading behavior-only training sessions into the BehaviorOphysSession object

        Args:
            behavior_session_id: the LIMS ID for the session. Not the mtrain session id.
        '''
        super().__init__(behavior_experiment_id)
        self.behavior_experiment_id = behavior_experiment_id

    @memoize
    def get_stimulus_timestamps(self):
        # We don't have a sync file, so we have to get vsync times from the pickle file
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        vsyncs = data["items"]["behavior"]["intervalsms"]
        return np.hstack((0, vsyncs)).cumsum() / 1000.0  # cumulative time

    @memoize
    def get_behavior_stimulus_file(self):
        # Use behavior_sessions table instead of ophys_sessions
        query = '''
                SELECT 
                stim.storage_directory || stim.filename AS stim_file
                FROM behavior_sessions bs
                LEFT JOIN well_known_files stim 
                ON stim.attachable_id=bs.id 
                AND stim.attachable_type = 'BehaviorSession' 
                AND stim.well_known_file_type_id IN
                (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
                WHERE bs.id={}
                '''.format(self.behavior_experiment_id)
        return safe_system_path(self.fetchone(query, strict=True))

    @memoize
    def get_licks(self):
        # Get licks from pickle file instead of sync
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        stimulus_timestamps = self.get_stimulus_timestamps()
        lick_frames = data['items']['behavior']['lick_sensors'][0]['lick_events']
        lick_times = [stimulus_timestamps[frame] for frame in lick_frames]
        return pd.DataFrame({'time': lick_times})

    def get_stimulus_rebase_function(self):
        # No sync times to rebase on, so just do nothing.
        return lambda x: x

    ## TODO: This is probably all going to break
    #  @memoize
    #  def get_metadata(self):
    #  
    #      metadata = super().get_metadata()
    #      metadata['ophys_experiment_id'] = self.get_ophys_experiment_id()
    #      metadata['experiment_container_id'] = self.get_experiment_container_id()
    #      metadata['ophys_frame_rate'] = self.get_ophys_frame_rate()
    #      metadata['stimulus_frame_rate'] = self.get_stimulus_frame_rate()
    #      metadata['targeted_structure'] = self.get_targeted_structure()
    #      metadata['imaging_depth'] = self.get_imaging_depth()
    #      metadata['session_type'] = self.get_stimulus_name()
    #      metadata['experiment_datetime'] = self.get_experiment_date()
    #      metadata['reporter_line'] = self.get_reporter_line()
    #      metadata['driver_line'] = self.get_driver_line()
    #      metadata['LabTracks_ID'] = self.get_external_specimen_name()
    #      metadata['full_genotype'] = self.get_full_genotype()
    #      metadata['behavior_session_uuid'] = uuid.UUID(self.get_behavior_session_uuid())
    #  
    #      return metadata

    ## TODO: Do we want to implement this for this class? 
    #  @memoize
    #  def get_nwb_filepath(self):
    #  
    #      query = '''
    #              SELECT wkf.storage_directory || wkf.filename AS nwb_file
    #              FROM ophys_experiments oe
    #              LEFT JOIN well_known_files wkf ON wkf.attachable_id=oe.id AND wkf.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'BehaviorOphysNwb')
    #              WHERE oe.id = {};
    #              '''.format(self.get_ophys_experiment_id())
    #      return safe_system_path(self.fetchone(query, strict=True))

    # TODO: This isn't working
    #  def get_extended_trials(self):
    #      filename = self.get_behavior_stimulus_file()
    #      data = pd.read_pickle(filename)
    #      return get_extended_trials(data)

if __name__ == "__main__":
    pass

    # print(BehaviorOphysLimsApi.get_ophys_experiment_df())
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
