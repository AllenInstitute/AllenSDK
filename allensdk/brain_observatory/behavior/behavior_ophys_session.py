import numpy as np
import pandas as pd
import math
from typing import NamedTuple
import os

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import equals
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate, dprime


class BehaviorOphysSession(LazyPropertyMixin):
    """Represents data from a single Visual Behavior Ophys imaging session.  LazyProperty attributes access the data only on the first demand, and then memoize the result for reuse.
    
    Attributes:
        ophys_experiment_id : int (LazyProperty)
            Unique identifier for this experimental session
        max_projection : SimpleITK.Image (LazyProperty)
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
        average_projection : SimpleITK.Image (LazyProperty)
            2D image of the microscope field of view, averaged across the experiment
        motion_correction : pandas.DataFrame LazyProperty
            A dataframe containing trace data used during motion correction computation
    """

    @classmethod
    def from_LIMS(cls, ophys_experiment_id):
        return cls(api=BehaviorOphysLimsApi(ophys_experiment_id))

    def __init__(self, api=None):

        self.api = api

        self.ophys_experiment_id = LazyProperty(self.api.get_ophys_experiment_id)
        self.max_projection = LazyProperty(self.api.get_max_projection)
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
        self.average_projection = LazyProperty(self.api.get_average_projection)
        self.motion_correction = LazyProperty(self.api.get_motion_correction)
        self.segmentation_mask_image = LazyProperty(self.api.get_segmentation_mask_image)

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
        performance_metrics = {}
        performance_metrics['reward_rate'] = self.get_reward_rate()

        hit = np.full(len(self.trials), np.nan)
        hit[self.trials.hit] = True
        hit[self.trials.miss] = False
        performance_metrics['hit_rate'] = pd.Series(hit).rolling(window=100, min_periods=0).mean()

        false_alarm = np.full(len(self.trials), np.nan)
        false_alarm[self.trials.false_alarm] = True
        false_alarm[self.trials.correct_reject] = False
        performance_metrics['false_alarm_rate'] = pd.Series(false_alarm).rolling(window=100, min_periods=0).mean()

        performance_metrics['dprime'] = dprime(performance_metrics['hit_rate'], performance_metrics['false_alarm_rate'])
        return pd.DataFrame(performance_metrics)

    def get_performance_metrics(self):
        performance_metrics = {}
        performance_metrics['trial_count'] = len(self.trials)
        performance_metrics['go_trial_count'] = self.trials.go.sum()
        performance_metrics['catch_trial_count'] = self.trials.catch.sum()
        performance_metrics['hit_trial_count'] = self.trials.hit.sum()
        performance_metrics['miss_trial_count'] = self.trials.miss.sum()
        performance_metrics['false_alarm_trial_count'] = self.trials.false_alarm.sum()
        performance_metrics['correct_reject_trial_count'] = self.trials.correct_reject.sum()
        performance_metrics['auto_rewarded_trial_count'] = self.trials.auto_rewarded.sum()
        performance_metrics['rewarded_trial_count'] = self.trials.reward_times.apply(lambda x: len(x) > 0).sum()
        performance_metrics['total_reward_count'] = len(self.rewards)
        performance_metrics['total_reward_volume'] = self.rewards.volume.sum()

        rolling_performance_df = self.get_rolling_performance_df()
        engaged_trial_mask = (rolling_performance_df['reward_rate'] > 2)
        performance_metrics['maximum_reward_rate'] = np.nanmax(rolling_performance_df['reward_rate'].values)
        performance_metrics['engaged_trial_count'] = (engaged_trial_mask).sum()
        performance_metrics['mean_false_alarm_rate'] = rolling_performance_df['false_alarm_rate'].mean()
        performance_metrics['mean_false_alarm_rate_engaged'] = rolling_performance_df['false_alarm_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_dprime'] = rolling_performance_df['dprime'].mean()
        performance_metrics['mean_dprime_engaged'] = rolling_performance_df['dprime'][engaged_trial_mask].mean()


        

        for key, val in performance_metrics.items():
            print(key, val)


if __name__ == "__main__":

    ophys_experiment_id = 789359614
    session = BehaviorOphysSession.from_LIMS(ophys_experiment_id)
    session.get_performance_metrics()

    # from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi

    # blacklist = [797257159, 796306435, 791453299, 809191721, 796308505, 798404219] #
    # api_list = []
    # df = BehaviorOphysLimsApi.get_ophys_experiment_df()
    # for cid in [791352433, 814796698, 814796612, 814796558, 814797528]:
    #     df2 = df[(df['container_id'] == cid) & (df['workflow_state'] == 'passed')]
    #     api_list += [BehaviorOphysLimsApi(oeid) for oeid in df2['ophys_experiment_id'].values if oeid not in blacklist]

    # for api in api_list:

    #     session = BehaviorOphysSession(api=api)
    #     if len(session.licks) > 100:

    #         print(api.get_ophys_experiment_id())

        # session.max_projection
        # session.stimulus_timestamps
        # session.ophys_timestamps
        # session.metadata
        # session.dff_traces
        # session.cell_specimen_table
        # session.running_speed
        # session.running_data_df

        # print(api.get_ophys_experiment_id(), len(session.licks), session.metadata['experiment_datetime'])
        # session.rewards
        # session.task_parameters
        # session.trials
        # session.corrected_fluorescence_traces
        # session.motion_correction

            # nwb_filepath = '/allen/aibs/technology/nicholasc/tmp/behavior_ophys_session_{get_ophys_experiment_id}.nwb'.format(get_ophys_experiment_id=api.get_ophys_experiment_id())
            # BehaviorOphysNwbApi(nwb_filepath).save(session)
            # assert equals(session, BehaviorOphysSession(api=BehaviorOphysNwbApi(nwb_filepath)))




        # print(session.running_speed)


    # nwb_filepath = '/home/nicholasc/projects/allensdk/tmp.nwb'
    # session = BehaviorOphysSession(789359614)
    # nwb_api = BehaviorOphysNwbApi(nwb_filepath)
    # nwb_api.save(session)

    # print(session.cell_specimen_table)

    # session2 = BehaviorOphysSession(789359614, api=api_2)
    
    # assert session == session2
    


    # ophys_experiment_id = 789359614

    # nwb_filepath = BehaviorOphysSession.from_LIMS(ophys_experiment_id).api.get_nwb_filepath()
    # api = BehaviorOphysNwbApi(nwb_filepath)
    # session = BehaviorOphysSession(api=api)




    # session = BehaviorOphysSession.from_LIMS(789359614)
    # BehaviorOphysNwbApi(nwb_filepath).save(session)
    # print(session.segmentation_mask_image)
    # session.stimulus_timestamps
    # session.ophys_timestamps
    # session.metadata
    # session.dff_traces
    # session.cell_specimen_table
    # running_speed
    # print(session.stimulus_index)
    # session.running_data_df
    # print(session.stimulus_presentations)
    # session.stimulus_templates
    # session.stimulus_index
    # session.licks
    # session.rewards
    # session.task_parameters
    # session.trials
    # session.corrected_fluorescence_traces
    # session.average_projection
    # session.motion_correction
