import numpy as np
import pandas as pd
import math
from typing import NamedTuple
import os

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import equals, BehaviorOphysNwbApi
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate
from allensdk.brain_observatory.behavior.dprime import get_rolling_dprime, get_trial_count_corrected_false_alarm_rate, get_trial_count_corrected_hit_rate
from allensdk.brain_observatory.behavior.dprime import get_hit_rate, get_false_alarm_rate
from allensdk.brain_observatory.behavior.image_api import ImageApi


class BehaviorOphysSession(LazyPropertyMixin):
    """Represents data from a single Visual Behavior Ophys imaging session.  LazyProperty attributes access the data only on the first demand, and then memoize the result for reuse.
    
    Attributes:
        ophys_experiment_id : int (LazyProperty)
            Unique identifier for this experimental session
        max_projection : allensdk.brain_observatory.behavior.image_api.Image (LazyProperty)
            2D max projection image
        stimulus_timestamps : numpy.ndarray (LazyProperty)
            Timestamps associated the stimulus presentations on the monitor 
        ophys_timestamps : numpy.ndarray (LazyProperty)
            Timestamps associated with frames captured by the microscope
        metadata : dict (LazyProperty)
            A dictionary of session-specific metadata
        dff_traces : pandas.DataFrame (LazyProperty)
            The traces of dff organized into a dataframe; index is the cell roi ids
        cell_specimen_table : pandas.DataFrame (LazyProperty)
            Cell roi information organized into a dataframe; index is the cell roi ids
        running_speed : allensdk.brain_observatory.running_speed.RunningSpeed (LazyProperty)
            NamedTuple with two fields
                timestamps : numpy.ndarray
                    Timestamps of running speed data samples
                values : np.ndarray
                    Running speed of the experimental subject (in cm / s).
        running_data_df : pandas.DataFrame (LazyProperty)
            Dataframe containing various signals used to compute running speed
        stimulus_presentations : pandas.DataFrame (LazyProperty)
            Table whose rows are stimulus presentations (i.e. a given image, for a given duration, typically 250 ms) and whose columns are presentation characteristics.
        stimulus_templates : dict (LazyProperty)
            A dictionary containing the stimulus images presented during the session keys are data set names, and values are 3D numpy arrays.
        licks : pandas.DataFrame (LazyProperty)
            A dataframe containing lick timestamps
        rewards : pandas.DataFrame (LazyProperty)
            A dataframe containing timestamps of delivered rewards
        task_parameters : dict (LazyProperty)
            A dictionary containing parameters used to define the task runtime behavior
        trials : pandas.DataFrame (LazyProperty)
            A dataframe containing behavioral trial start/stop times, and trial data
        corrected_fluorescence_traces : pandas.DataFrame (LazyProperty)
            The motion-corrected fluorescence traces organized into a dataframe; index is the cell roi ids
        average_projection : allensdk.brain_observatory.behavior.image_api.Image (LazyProperty)
            2D image of the microscope field of view, averaged across the experiment
        motion_correction : pandas.DataFrame LazyProperty
            A dataframe containing trace data used during motion correction computation
    """

    @classmethod
    def from_lims(cls, ophys_experiment_id):
        return cls(api=BehaviorOphysLimsApi(ophys_experiment_id))

    @classmethod
    def from_nwb_path(cls, nwb_path, **api_kwargs):
        api_kwargs["filter_invalid_rois"] = api_kwargs.get("filter_invalid_rois", True)
        return cls(api=BehaviorOphysNwbApi.from_path(path=nwb_path, **api_kwargs))

    def __init__(self, api=None):

        self.api = api

        self.ophys_experiment_id = LazyProperty(self.api.get_ophys_experiment_id)
        self.max_projection = LazyProperty(self.get_max_projection)
        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps)
        self.metadata = LazyProperty(self.api.get_metadata)
        self.dff_traces = LazyProperty(self.api.get_dff_traces)
        self.cell_specimen_table = LazyProperty(self.api.get_cell_specimen_table)
        self.running_speed = LazyProperty(self.api.get_running_speed)
        self.running_data_df = LazyProperty(self.api.get_running_data_df)
        self.stimulus_presentations = LazyProperty(self.api.get_stimulus_presentations)
        self.stimulus_templates = LazyProperty(self.api.get_stimulus_templates)
        self.licks = LazyProperty(self.api.get_licks)
        self.rewards = LazyProperty(self.api.get_rewards)
        self.task_parameters = LazyProperty(self.api.get_task_parameters)
        self.trials = LazyProperty(self.api.get_trials)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces)
        self.average_projection = LazyProperty(self.get_average_projection)
        self.motion_correction = LazyProperty(self.api.get_motion_correction)
        self.segmentation_mask_image = LazyProperty(self.get_segmentation_mask_image)

    @legacy('Consider using "get_dff_timeseries" instead.')
    def get_dff_traces(self, cell_specimen_ids=None):

        if cell_specimen_ids is None:
            cell_specimen_ids = self.get_cell_specimen_ids()

        csid_table = self.cell_specimen_table.reset_index()[['cell_specimen_id']]
        csid_subtable = csid_table[csid_table['cell_specimen_id'].isin(cell_specimen_ids)].set_index('cell_specimen_id')
        dff_table = csid_subtable.join(self.dff_traces, how='left')
        dff_traces = np.vstack(dff_table['dff'].values)
        timestamps = self.ophys_timestamps

        assert (len(cell_specimen_ids), len(timestamps)) == dff_traces.shape
        return timestamps, dff_traces

    @legacy()
    def get_cell_specimen_indices(self, cell_specimen_ids):
        return [self.cell_specimen_table.index.get_loc(csid) for csid in cell_specimen_ids]

    @legacy('Consider using "cell_specimen_table[\'cell_specimen_id\']" instead.')
    def get_cell_specimen_ids(self):
        cell_specimen_ids = self.cell_specimen_table.index.values

        if np.isnan(cell_specimen_ids.astype(float)).sum() == len(self.cell_specimen_table):
            raise ValueError(f'cell_specimen_id values not assigned for {self.ophys_experiment_id}')
        return cell_specimen_ids

    def deserialize_image(self, sitk_image):
        '''
        Convert SimpleITK image returned by the api to an Image class:

        Args:
            sitk_image (SimpleITK image): image object returned by the api

        Returns
            img (allensdk.brain_observatory.behavior.image_api.Image)
        '''
        img = ImageApi.deserialize(sitk_image)
        return img

    def get_max_projection(self):
        """ Returns an image whose values are the maximum obtained values at each pixel of the ophys movie over time.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to max projection image data and metadata
        """
        return self.deserialize_image(self.api.get_max_projection())

    def get_average_projection(self):
        """ Returns an image whose values are the average obtained values at each pixel of the ophys movie over time.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to max projection image data and metadata
        """
        return self.deserialize_image(self.api.get_average_projection())

    def get_segmentation_mask_image(self):
        """ Returns an image with a pixel value of zero if the pixel is not included in any ROI, and nonzero if included in a segmented ROI.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to max projection image data and metadata
        """
        return self.deserialize_image(self.api.get_segmentation_mask_image())

    def get_reward_rate(self):
        response_latency_list = []
        for _, t in self.trials.iterrows():
            valid_response_licks = [l for l in t.lick_times if l - t.change_time > self.task_parameters['response_window_sec'][0]]
            response_latency = float('inf') if len(valid_response_licks) == 0 else valid_response_licks[0] - t.change_time
            response_latency_list.append(response_latency)
        reward_rate = calculate_reward_rate(response_latency=response_latency_list, starttime=self.trials.start_time.values)
        reward_rate[np.isinf(reward_rate)] = float('nan')
        return reward_rate

    def get_rolling_performance_df(self):

        # Indices to build trial metrics dataframe:
        trials_index = self.trials.index
        not_aborted_index = self.trials[np.logical_not(self.trials.aborted)].index

        # Initialize dataframe:
        performance_metrics_df = pd.DataFrame(index=trials_index)

        # Reward rate:
        performance_metrics_df['reward_rate'] = pd.Series(self.get_reward_rate(), index=self.trials.index)

        # Hit rate raw:
        hit_rate_raw = get_hit_rate(hit=self.trials.hit, miss=self.trials.miss, aborted=self.trials.aborted)
        performance_metrics_df['hit_rate_raw'] = pd.Series(hit_rate_raw, index=not_aborted_index)
        
        # Hit rate with trial count correction:
        hit_rate = get_trial_count_corrected_hit_rate(hit=self.trials.hit, miss=self.trials.miss, aborted=self.trials.aborted)
        performance_metrics_df['hit_rate'] = pd.Series(hit_rate, index=not_aborted_index)

        # False-alarm rate raw:
        false_alarm_rate_raw = get_false_alarm_rate(false_alarm=self.trials.false_alarm, correct_reject=self.trials.correct_reject, aborted=self.trials.aborted)
        performance_metrics_df['false_alarm_rate_raw'] = pd.Series(false_alarm_rate_raw, index=not_aborted_index)
        
        # False-alarm rate with trial count correction:
        false_alarm_rate = get_trial_count_corrected_false_alarm_rate(false_alarm=self.trials.false_alarm, correct_reject=self.trials.correct_reject, aborted=self.trials.aborted)
        performance_metrics_df['false_alarm_rate'] = pd.Series(false_alarm_rate, index=not_aborted_index)

        # Rolling-dprime:
        rolling_dprime = get_rolling_dprime(hit_rate, false_alarm_rate)
        performance_metrics_df['rolling_dprime'] = pd.Series(rolling_dprime, index=not_aborted_index)

        return performance_metrics_df

    def get_performance_metrics(self, engaged_trial_reward_rate_threshold=2):
        performance_metrics = {}
        performance_metrics['trial_count'] = len(self.trials)
        performance_metrics['go_trial_count'] = self.trials.go.sum()
        performance_metrics['catch_trial_count'] = self.trials.catch.sum()
        performance_metrics['hit_trial_count'] = self.trials.hit.sum()
        performance_metrics['miss_trial_count'] = self.trials.miss.sum()
        performance_metrics['false_alarm_trial_count'] = self.trials.false_alarm.sum()
        performance_metrics['correct_reject_trial_count'] = self.trials.correct_reject.sum()
        performance_metrics['auto_rewarded_trial_count'] = self.trials.auto_rewarded.sum()
        performance_metrics['rewarded_trial_count'] = self.trials.reward_time.apply(lambda x: not np.isnan(x)).sum()
        performance_metrics['total_reward_count'] = len(self.rewards)
        performance_metrics['total_reward_volume'] = self.rewards.volume.sum()

        rolling_performance_df = self.get_rolling_performance_df()
        engaged_trial_mask = (rolling_performance_df['reward_rate'] > engaged_trial_reward_rate_threshold)
        performance_metrics['maximum_reward_rate'] = np.nanmax(rolling_performance_df['reward_rate'].values)
        performance_metrics['engaged_trial_count'] = (engaged_trial_mask).sum()
        performance_metrics['mean_hit_rate'] = rolling_performance_df['hit_rate'].mean()
        performance_metrics['mean_hit_rate_uncorrected'] = rolling_performance_df['hit_rate_raw'].mean()
        performance_metrics['mean_hit_rate_engaged'] = rolling_performance_df['hit_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_false_alarm_rate'] = rolling_performance_df['false_alarm_rate'].mean()
        performance_metrics['mean_false_alarm_rate_uncorrected'] = rolling_performance_df['false_alarm_rate_raw'].mean()
        performance_metrics['mean_false_alarm_rate_engaged'] = rolling_performance_df['false_alarm_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_dprime'] = rolling_performance_df['rolling_dprime'].mean()
        performance_metrics['mean_dprime_engaged'] = rolling_performance_df['rolling_dprime'][engaged_trial_mask].mean()
        performance_metrics['max_dprime'] = rolling_performance_df['rolling_dprime'].mean()
        performance_metrics['max_dprime_engaged'] = rolling_performance_df['rolling_dprime'][engaged_trial_mask].max()

        return performance_metrics

if __name__ == "__main__":

    ophys_experiment_id = 789359614
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)
    print(session.trials['reward_time'])
