import numpy as np
import pandas as pd
import math
from typing import NamedTuple
import os

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import equals
from allensdk.internal.api.behavior_only_api import BehaviorOnlyLimsApi
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.trials_processing import calculate_reward_rate
from allensdk.brain_observatory.behavior.dprime import get_trial_count_corrected_hit_rate, get_trial_count_corrected_false_alarm_rate, get_rolling_dprime


class BehaviorSession(LazyPropertyMixin):
    
    """Represents data from a single Visual Behavior session. LazyProperty attributes access the data only on the first demand, and then memoize the result for reuse.
    
    Attributes:
        behavior_session_id : int (LazyProperty)
            Unique identifier for this experimental session
        stimulus_timestamps : numpy.ndarray (LazyProperty)
            Timestamps associated the stimulus presentations on the monitor 
        metadata : dict (LazyProperty)
            A dictionary of session-specific metadata
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
    """

    def __init__(self, api=None):
        self.api = api

        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps)
        self.metadata = LazyProperty(self.api.get_metadata)
        self.running_speed = LazyProperty(self.api.get_running_speed)
        self.running_data_df = LazyProperty(self.api.get_running_data_df)
        self.stimulus_presentations = LazyProperty(self.api.get_stimulus_presentations)
        self.stimulus_templates = LazyProperty(self.api.get_stimulus_templates)
        self.licks = LazyProperty(self.api.get_licks)
        self.rewards = LazyProperty(self.api.get_rewards)
        self.task_parameters = LazyProperty(self.api.get_task_parameters)
        self.trials = LazyProperty(self.api.get_trials)

    @classmethod
    def from_lims(cls, behavior_session_id):
        return cls(api=BehaviorOnlyLimsApi(behavior_session_id))

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

        # Hit rate:
        hit_rate = get_trial_count_corrected_hit_rate(hit=self.trials.hit, miss=self.trials.miss, aborted=self.trials.aborted)
        performance_metrics_df['hit_rate'] = pd.Series(hit_rate, index=not_aborted_index)

        # False-alarm rate:
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
        performance_metrics['rewarded_trial_count'] = self.trials.reward_times.apply(lambda x: len(x) > 0).sum()
        performance_metrics['total_reward_count'] = len(self.rewards)
        performance_metrics['total_reward_volume'] = self.rewards.volume.sum()

        rolling_performance_df = self.get_rolling_performance_df()
        engaged_trial_mask = (rolling_performance_df['reward_rate'] > engaged_trial_reward_rate_threshold)
        performance_metrics['maximum_reward_rate'] = np.nanmax(rolling_performance_df['reward_rate'].values)
        performance_metrics['engaged_trial_count'] = (engaged_trial_mask).sum()
        performance_metrics['mean_hit_rate'] = rolling_performance_df['hit_rate'].mean()
        performance_metrics['mean_hit_rate_engaged'] = rolling_performance_df['hit_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_false_alarm_rate'] = rolling_performance_df['false_alarm_rate'].mean()
        performance_metrics['mean_false_alarm_rate_engaged'] = rolling_performance_df['false_alarm_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_dprime'] = rolling_performance_df['rolling_dprime'].mean()
        performance_metrics['mean_dprime_engaged'] = rolling_performance_df['rolling_dprime'][engaged_trial_mask].mean()
        performance_metrics['max_dprime'] = rolling_performance_df['rolling_dprime'].mean()
        performance_metrics['max_dprime_engaged'] = rolling_performance_df['rolling_dprime'][engaged_trial_mask].max()

        return performance_metrics


if __name__=="__main__":

    # Behavior session from donor 842724844 on date 2019-04-26
    behavior_session_id = 858098100
    api = BehaviorOnlyLimsApi(behavior_session_id)
    session = BehaviorSession(api)
    print(session.get_performance_metrics())
