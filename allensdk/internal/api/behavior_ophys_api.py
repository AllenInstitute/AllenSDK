import matplotlib.image as mpimg  # NOQA: E402
import numpy as np
import h5py
import pandas as pd
import uuid

from allensdk.api.cache import memoize
from allensdk.internal.api.ophys_lims_api import OphysLimsApi
from allensdk.brain_observatory.behavior.sync import get_sync_data, get_stimulus_rebase_function
from allensdk.brain_observatory.behavior.roi_processing import get_roi_metrics
from allensdk.brain_observatory.behavior.stimulus_processing import get_stimtable, get_stimulus_template, get_stimulus_metadata
from allensdk.brain_observatory.behavior.metadata_processing import get_task_parameters
from allensdk.brain_observatory.behavior.running_processing import get_running_df
from allensdk.brain_observatory.behavior.rewards_processing import get_rewards
from allensdk.brain_observatory.behavior.trials_processing import get_trials
from allensdk.brain_observatory import RunningSpeed


class BehaviorOphysLimsApi(OphysLimsApi):

    @memoize
    def get_sync_data(self, ophys_experiment_id=None, use_acq_trigger=False):
        sync_path = self.get_sync_file(ophys_experiment_id=ophys_experiment_id)
        return get_sync_data(sync_path, use_acq_trigger=use_acq_trigger)


    @memoize
    def get_stimulus_timestamps(self, ophys_experiment_id=None, use_acq_trigger=False):
        return self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['stimulus_frames']


    @memoize
    def get_ophys_timestamps(self, ophys_experiment_id=None, use_acq_trigger=False):
        return self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['ophys_frames']


    @memoize
    def get_experiment_container_id(self, ophys_experiment_id=None):
        query = '''
                SELECT visual_behavior_experiment_container_id 
                FROM ophys_experiments_visual_behavior_experiment_containers 
                WHERE ophys_experiment_id= {};
                '''.format(ophys_experiment_id)        
        return self.fetchone(query, strict=False)


    @memoize
    def get_behavior_stimulus_file(self, ophys_experiment_id=None):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs ON bs.ophys_session_id=os.id
                LEFT JOIN well_known_files stim ON stim.attachable_id=bs.id AND stim.attachable_type = 'BehaviorSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
                WHERE oe.id= {};
                '''.format(ophys_experiment_id)
        return self.fetchone(query, strict=True)


    def get_behavior_session_uuid(self, ophys_experiment_id=None):
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return data['session_uuid']

    @memoize
    def get_stimulus_frame_rate(self, ophys_experiment_id=None, use_acq_trigger=False):
        stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        return np.round(1 / np.mean(np.diff(stimulus_timestamps)), 0)


    @memoize
    def get_ophys_frame_rate(self, ophys_experiment_id=None, use_acq_trigger=False):
        ophys_timestamps = self.get_ophys_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        return np.round(1 / np.mean(np.diff(ophys_timestamps)), 0)


    @memoize
    def get_metadata(self, ophys_experiment_id=None, use_acq_trigger=False):

        metadata = {}
        metadata['ophys_experiment_id'] = ophys_experiment_id
        metadata['experiment_container_id'] = self.get_experiment_container_id(ophys_experiment_id=ophys_experiment_id)
        metadata['ophys_frame_rate'] = self.get_ophys_frame_rate(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        metadata['stimulus_frame_rate'] = self.get_stimulus_frame_rate(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        metadata['targeted_structure'] = self.get_targeted_structure(ophys_experiment_id)
        metadata['imaging_depth'] = self.get_imaging_depth(ophys_experiment_id)
        metadata['session_type'] = self.get_stimulus_name(ophys_experiment_id)
        metadata['experiment_date'] = self.get_experiment_date(ophys_experiment_id)
        metadata['reporter_line'] = self.get_reporter_line(ophys_experiment_id)
        metadata['driver_line'] = self.get_driver_line(ophys_experiment_id)
        metadata['LabTracks_ID'] = self.get_LabTracks_ID(ophys_experiment_id)
        metadata['full_genotype'] = self.get_full_genotype(ophys_experiment_id)
        metadata['behavior_session_uuid'] = uuid.UUID(self.get_behavior_session_uuid(ophys_experiment_id))
        metadata['rig'] = self.get_equipment_id(ophys_experiment_id)

        return metadata

    @memoize
    def get_dff_traces(self, ophys_experiment_id=None, use_acq_trigger=False):
        dff_traces = self.get_raw_dff_data(ophys_experiment_id)
        cell_roi_id_list = self.get_cell_roi_ids(ophys_experiment_id=ophys_experiment_id)
        df = pd.DataFrame({'cell_roi_id':cell_roi_id_list, 'dff':list(dff_traces)})
        return df

    @memoize
    def get_roi_metrics(self, ophys_experiment_id=None):
        input_extract_traces_file = self.get_input_extract_traces_file(ophys_experiment_id=ophys_experiment_id)
        objectlist_file = self.get_objectlist_file(ophys_experiment_id=ophys_experiment_id)
        return get_roi_metrics(input_extract_traces_file, ophys_experiment_id, objectlist_file)['unfiltered']

    @memoize
    def get_running_data_df(self, ophys_experiment_id=None, use_acq_trigger=False):
        stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return get_running_df(data, stimulus_timestamps)

    @memoize
    def get_running_speed(self, ophys_experiment_id=None, use_acq_trigger=False):
        running_data_df = self.get_running_data_df(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        assert running_data_df.index.name == 'timestamps'
        return RunningSpeed(timestamps=running_data_df.index.values,
                            values=running_data_df.speed.values)


    @memoize    
    def get_stimulus_table(self, ophys_experiment_id=None, use_acq_trigger=False):
        stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return get_stimtable(data, stimulus_timestamps)


    @memoize
    def get_stimulus_template(self, ophys_experiment_id=None):
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return get_stimulus_template(data)


    @memoize
    def get_stimulus_metadata(self, ophys_experiment_id=None):
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return get_stimulus_metadata(data)


    @memoize
    def get_licks(self, ophys_experiment_id=None, use_acq_trigger=False):
        lick_times = self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['lick_times']
        return pd.DataFrame(data={"time": lick_times, })


    @memoize
    def get_rewards(self, ophys_experiment_id=None, use_acq_trigger=False):
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        stimulus_timestamps = self.get_stimulus_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        data = pd.read_pickle(behavior_stimulus_file)
        rebase_function = self.get_stimulus_rebase_function(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        return get_rewards(data, stimulus_timestamps, rebase_function)


    @memoize
    def get_task_parameters(self, ophys_experiment_id=None):
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        return get_task_parameters(data)


    @memoize
    def get_trials(self, ophys_experiment_id=None, use_acq_trigger=False):

        stimulus_timestamps_no_monitor_delay = self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['stimulus_frames_no_delay']
        licks = self.get_licks(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        rewards = self.get_rewards(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        rebase_function = self.get_stimulus_rebase_function(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)

        trial_df = get_trials(data, stimulus_timestamps_no_monitor_delay, licks, rewards, rebase_function)

        return trial_df


    @memoize
    def get_corrected_fluorescence_traces(self, ophys_experiment_id=None, use_acq_trigger=False):
        demix_file = self.get_demix_file(ophys_experiment_id=ophys_experiment_id)
        
        g = h5py.File(demix_file)
        corrected_fluorescence_trace_array = np.asarray(g['data'])
        g.close()

        cell_roi_id_list = self.get_cell_roi_ids(ophys_experiment_id=ophys_experiment_id)
        ophys_timestamps = self.get_ophys_timestamps(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)
        assert corrected_fluorescence_trace_array.shape[1], ophys_timestamps.shape[0]
        df = pd.DataFrame({'roi_id':cell_roi_id_list, 'corrected_fluorescence':list(corrected_fluorescence_trace_array)})
        return df


    @memoize
    def get_average_image(self, ophys_experiment_id=None):
        avgint_a1X_file = self.get_avgint_a1X_file(ophys_experiment_id=ophys_experiment_id)
        average_image = mpimg.imread(avgint_a1X_file)
        return average_image


    @memoize
    def get_motion_correction(self, ophys_experiment_id=None):
        motion_correction_filepath = self.get_rigid_motion_transform_file(ophys_experiment_id=ophys_experiment_id)
        motion_correction = pd.read_csv(motion_correction_filepath)
        return motion_correction


    def get_stimulus_rebase_function(self, ophys_experiment_id=None, use_acq_trigger=False):

        stimulus_timestamps_no_monitor_delay = self.get_sync_data(ophys_experiment_id=ophys_experiment_id, use_acq_trigger=use_acq_trigger)['stimulus_frames_no_delay']
        behavior_stimulus_file = self.get_behavior_stimulus_file(ophys_experiment_id=ophys_experiment_id)
        data = pd.read_pickle(behavior_stimulus_file)
        stimulus_rebase_function = get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay)

        return stimulus_rebase_function