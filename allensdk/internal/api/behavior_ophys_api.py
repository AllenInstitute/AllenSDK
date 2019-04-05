import matplotlib.image as mpimg  # NOQA: E402
import numpy as np
import h5py
import pandas as pd
import uuid
import json

from allensdk.api.cache import memoize
from allensdk.internal.api.ophys_lims_api import OphysLimsApi
from allensdk.brain_observatory.behavior.sync import get_sync_data, get_stimulus_rebase_function
from allensdk.brain_observatory.behavior.stimulus_processing import get_stimulus_presentations, get_stimulus_templates, get_stimulus_metadata
from allensdk.brain_observatory.behavior.metadata_processing import get_task_parameters
from allensdk.brain_observatory.behavior.running_processing import get_running_df
from allensdk.brain_observatory.behavior.rewards_processing import get_rewards
from allensdk.brain_observatory.behavior.trials_processing import get_trials
from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.brain_observatory.behavior.image_api import ImageApi


class BehaviorOphysLimsApi(OphysLimsApi):

    def __init__(self, ophys_experiment_id):
        super().__init__(ophys_experiment_id)

    @memoize
    def get_sync_data(self):
        sync_path = self.get_sync_file()
        return get_sync_data(sync_path)

    @memoize
    def get_stimulus_timestamps(self):
        return self.get_sync_data()['stimulus_frames']


    @memoize
    def get_ophys_timestamps(self):
        return self.get_sync_data()['ophys_frames']

    @memoize
    def get_experiment_container_id(self):
        query = '''
                SELECT visual_behavior_experiment_container_id 
                FROM ophys_experiments_visual_behavior_experiment_containers 
                WHERE ophys_experiment_id= {};
                '''.format(self.ophys_experiment_id)        
        return self.fetchone(query, strict=False)

    @memoize
    def get_behavior_stimulus_file(self):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs ON bs.ophys_session_id=os.id
                LEFT JOIN well_known_files stim ON stim.attachable_id=bs.id AND stim.attachable_type = 'BehaviorSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
                WHERE oe.id= {};
                '''.format(self.ophys_experiment_id)
        return self.fetchone(query, strict=True)

    def get_behavior_session_uuid(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return data['session_uuid']

    @memoize
    def get_stimulus_frame_rate(self):
        stimulus_timestamps = self.get_stimulus_timestamps()
        return np.round(1 / np.mean(np.diff(stimulus_timestamps)), 0)

    @memoize
    def get_ophys_frame_rate(self):
        ophys_timestamps = self.get_ophys_timestamps()
        return np.round(1 / np.mean(np.diff(ophys_timestamps)), 0)

    @memoize
    def get_metadata(self):

        metadata = super().get_metadata()
        metadata['ophys_experiment_id'] = self.ophys_experiment_id
        metadata['experiment_container_id'] = self.get_experiment_container_id()
        metadata['ophys_frame_rate'] = self.get_ophys_frame_rate()
        metadata['stimulus_frame_rate'] = self.get_stimulus_frame_rate()
        metadata['targeted_structure'] = self.get_targeted_structure()
        metadata['imaging_depth'] = self.get_imaging_depth()
        metadata['session_type'] = self.get_stimulus_name()
        metadata['experiment_datetime'] = self.get_experiment_date()
        metadata['reporter_line'] = self.get_reporter_line()
        metadata['driver_line'] = self.get_driver_line()
        metadata['LabTracks_ID'] = self.get_LabTracks_ID()
        metadata['full_genotype'] = self.get_full_genotype()
        metadata['behavior_session_uuid'] = uuid.UUID(self.get_behavior_session_uuid())

        return metadata

    @memoize
    def get_dff_traces(self):
        dff_traces = self.get_raw_dff_data()
        cell_roi_id_list = self.get_cell_roi_ids()
        df = pd.DataFrame({'dff': list(dff_traces)}, index=pd.Index(cell_roi_id_list, name='cell_roi_id'))
        return df

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
        return get_stimulus_presentations(data, stimulus_timestamps)

    @memoize
    def get_stimulus_templates(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return get_stimulus_templates(data)

    @memoize
    def get_stimulus_index(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        stimulus_metadata_df = get_stimulus_metadata(data)
        stimulus_presentations_df = self.get_stimulus_presentations()
        idx_name = stimulus_presentations_df.index.name
        stimulus_index_df = stimulus_presentations_df.reset_index().merge(stimulus_metadata_df.reset_index(), on=['image_name', 'image_category']).set_index(idx_name)
        stimulus_index_df.sort_index(inplace=True)
        stimulus_index_df = stimulus_index_df[['image_set', 'image_index', 'start_time']].rename(columns={'start_time': 'timestamps'})
        stimulus_index_df.set_index('timestamps', inplace=True, drop=True)
        return stimulus_index_df

    @memoize
    def get_licks(self):
        lick_times = self.get_sync_data()['lick_times']
        return pd.DataFrame({'time': lick_times})

    @memoize
    def get_rewards(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        stimulus_timestamps = self.get_stimulus_timestamps()
        data = pd.read_pickle(behavior_stimulus_file)
        rebase_function = self.get_stimulus_rebase_function()
        return get_rewards(data, stimulus_timestamps, rebase_function)

    @memoize
    def get_task_parameters(self):
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        return get_task_parameters(data)

    @memoize
    def get_trials(self):

        stimulus_timestamps_no_monitor_delay = self.get_sync_data()['stimulus_frames_no_delay']
        licks = self.get_licks()
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        rewards = self.get_rewards()
        rebase_function = self.get_stimulus_rebase_function()

        trial_df = get_trials(data, stimulus_timestamps_no_monitor_delay, licks, rewards, rebase_function)

        return trial_df

    @memoize
    def get_corrected_fluorescence_traces(self):
        demix_file = self.get_demix_file()
        
        g = h5py.File(demix_file)
        corrected_fluorescence_trace_array = np.asarray(g['data'])
        g.close()

        cell_roi_id_list = self.get_cell_roi_ids()
        ophys_timestamps = self.get_ophys_timestamps()
        assert corrected_fluorescence_trace_array.shape[1], ophys_timestamps.shape[0]
        df = pd.DataFrame({'corrected_fluorescence': list(corrected_fluorescence_trace_array)}, index=pd.Index(cell_roi_id_list, name='cell_roi_id'))
        return df

    @memoize
    def get_average_image(self, image_api=None):

        if image_api is None:
            image_api = ImageApi

        avgint_a1X_file = self.get_avgint_a1X_file()
        pixel_size = self.get_surface_2p_pixel_size_um()
        average_image = mpimg.imread(avgint_a1X_file)
        return ImageApi.serialize(average_image, [pixel_size / 1000., pixel_size / 1000.], 'mm')

    @memoize
    def get_motion_correction(self):
        motion_correction_filepath = self.get_rigid_motion_transform_file()
        motion_correction = pd.read_csv(motion_correction_filepath)
        return motion_correction[['x', 'y']]

    def get_stimulus_rebase_function(self):

        stimulus_timestamps_no_monitor_delay = self.get_sync_data()['stimulus_frames_no_delay']
        behavior_stimulus_file = self.get_behavior_stimulus_file()
        data = pd.read_pickle(behavior_stimulus_file)
        stimulus_rebase_function = get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay)

        return stimulus_rebase_function