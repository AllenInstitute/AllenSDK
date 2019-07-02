import numpy as np
import pandas as pd
import uuid

from allensdk.api.cache import memoize
from allensdk.brain_observatory.behavior.stimulus_processing import get_stimulus_presentations, get_stimulus_templates, get_stimulus_metadata
from allensdk.brain_observatory.behavior.metadata_processing import get_task_parameters
from allensdk.brain_observatory.behavior.running_processing import get_running_df
from allensdk.brain_observatory.behavior.rewards_processing import get_rewards
from allensdk.brain_observatory.behavior.trials_processing import get_trials
from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.brain_observatory.behavior.trials_processing import get_extended_trials
from allensdk.internal.core import lims_utilities
from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.core.lims_utilities import safe_system_path


class BehaviorLimsApi(PostgresQueryMixin):

    def __init__(self, behavior_experiment_id):
        """
        Notes
        -----
        - behavior_experiment_id is the same as behavior_session_id which is in lims
        - behavior_experiment_id is associated with foraging_id in lims
        - foraging_id in lims is the same as behavior_session_uuid in mtrain which is the same
        as session_uuid in the pickle returned by behavior_stimulus_file
        """
        self.behavior_experiment_id = behavior_experiment_id
        PostgresQueryMixin.__init__(self)

    def get_behavior_experiment_id(self):
        return self.behavior_experiment_id

    def get_behavior_session_uuid(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return data['session_uuid']

    @memoize
    def get_behavior_stimulus_file(self):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file 
                FROM behavior_sessions bs 
                LEFT JOIN well_known_files stim ON stim.attachable_id=bs.id AND stim.attachable_type = 'BehaviorSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle') 
                WHERE bs.id= {};
                '''.format(self.get_behavior_experiment_id())
        return safe_system_path(self.fetchone(query, strict=True))

    def get_extended_trials(self):
        filename = self.get_behavior_stimulus_file()
        data = pd.read_pickle(filename)
        return get_extended_trials(data)

    @staticmethod
    def foraging_id_to_behavior_session_id(foraging_id):
        '''maps foraging_id to behavior_session_id'''
        api = PostgresQueryMixin()
        query = '''select id from behavior_sessions where foraging_id = '{}';'''.format(
            foraging_id)
        return api.fetchone(query, strict=True)

    @staticmethod
    def behavior_session_id_to_foraging_id(behavior_session_id):
        '''maps behavior_session_id to foraging_id'''
        api = PostgresQueryMixin()
        query = '''select foraging_id from behavior_sessions where id = '{}';'''.format(
            behavior_session_id)
        return api.fetchone(query, strict=True)

    @classmethod
    def from_foraging_id(cls, foraging_id):
        return cls(
            behavior_experiment_id=cls.foraging_id_to_behavior_session_id(foraging_id),
        )

    @memoize
    def get_stimulus_timestamps(self):
        # We don't have a sync file, so we have to get vsync times from the pickle file
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        vsyncs = data["items"]["behavior"]["intervalsms"]
        return np.hstack((0, vsyncs)).cumsum() / 1000.0  # cumulative time

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

    @memoize
    def get_stimulus_frame_rate(self):
        stimulus_timestamps = self.get_stimulus_timestamps()
        return np.round(1 / np.mean(np.diff(stimulus_timestamps)), 0)

    @memoize
    def get_running_data_df(self):
        stimulus_timestamps = self.get_stimulus_timestamps()
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return get_running_df(data, stimulus_timestamps)

    @memoize
    def get_running_speed(self):
        running_data_df = self.get_running_data_df()
        assert running_data_df.index.name == 'timestamps'
        return RunningSpeed(timestamps=running_data_df.index.values,
                            values=running_data_df.speed.values)

    @memoize
    def get_stimulus_presentations(self):
        stimulus_timestamps = self.get_stimulus_timestamps()
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        stimulus_presentations_df_pre = get_stimulus_presentations(data, stimulus_timestamps)
        stimulus_metadata_df = get_stimulus_metadata(data)
        idx_name = stimulus_presentations_df_pre.index.name
        stimulus_index_df = stimulus_presentations_df_pre.reset_index().merge(stimulus_metadata_df.reset_index(), on=['image_name']).set_index(idx_name)
        stimulus_index_df.sort_index(inplace=True)
        stimulus_index_df = stimulus_index_df[['image_set', 'image_index', 'start_time']].rename(columns={'start_time': 'timestamps'})
        stimulus_index_df.set_index('timestamps', inplace=True, drop=True)
        stimulus_presentations_df = stimulus_presentations_df_pre.merge(stimulus_index_df, left_on='start_time', right_index=True, how='left')
        assert len(stimulus_presentations_df_pre) == len(stimulus_presentations_df)

        return stimulus_presentations_df[sorted(stimulus_presentations_df.columns)]

    @memoize
    def get_stimulus_templates(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return get_stimulus_templates(data)

    @memoize
    def get_rewards(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        rebase_function = self.get_stimulus_rebase_function()
        return get_rewards(data, rebase_function)

    @memoize
    def get_task_parameters(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return get_task_parameters(data)

    @memoize
    def get_trials(self):

        licks = self.get_licks()
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        rewards = self.get_rewards()
        rebase_function = self.get_stimulus_rebase_function()
        trial_df = get_trials(data, licks, rewards, rebase_function)

        return trial_df

    @memoize
    def get_metadata(self):

        metadata = {}
        metadata['behavior_experiment_id'] = self.get_behavior_experiment_id()
        # metadata['experiment_container_id'] = self.get_experiment_container_id()
        # metadata['ophys_frame_rate'] = self.get_ophys_frame_rate()
        metadata['stimulus_frame_rate'] = self.get_stimulus_frame_rate()
        # metadata['targeted_structure'] = self.get_targeted_structure()
        # metadata['imaging_depth'] = self.get_imaging_depth()
        # metadata['session_type'] = self.get_stimulus_name()
        # metadata['experiment_datetime'] = self.get_experiment_date()
        # metadata['reporter_line'] = self.get_reporter_line()
        # metadata['driver_line'] = self.get_driver_line()
        # metadata['LabTracks_ID'] = self.get_external_specimen_name()
        # metadata['full_genotype'] = self.get_full_genotype()
        metadata['behavior_session_uuid'] = uuid.UUID(self.get_behavior_session_uuid())

        return metadata

if __name__=="__main__":
    api = BehaviorLimsApi(858098100)
    print(api.get_trials())
