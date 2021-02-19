from typing import Any, Optional, List, Dict, Type, Tuple
import logging
import pandas as pd
import numpy as np
import inspect

from allensdk.brain_observatory.behavior.stimulus_processing import \
    StimulusTemplate
from allensdk.core.lazy_property import LazyPropertyMixin
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsApi, BehaviorNwbApi)
from allensdk.brain_observatory.behavior.session_apis.abcs import BehaviorBase
from allensdk.brain_observatory.behavior.trials_processing import (
    calculate_reward_rate)
from allensdk.brain_observatory.behavior.dprime import (
    get_rolling_dprime, get_trial_count_corrected_false_alarm_rate,
    get_trial_count_corrected_hit_rate)
from allensdk.brain_observatory.behavior.dprime import (
    get_hit_rate, get_false_alarm_rate)


BehaviorDataApi = Type[BehaviorBase]


class BehaviorSession(LazyPropertyMixin):
    def __init__(self, api: Optional[BehaviorDataApi] = None):
        self.api = api

        # LazyProperty constructor provided by LazyPropertyMixin
        LazyProperty = self.LazyProperty

        # Initialize attributes to be lazily evaluated
        self._behavior_session_id = LazyProperty(
            self.api.get_behavior_session_id)
        self._licks = LazyProperty(self.api.get_licks, settable=True)
        self._rewards = LazyProperty(self.api.get_rewards, settable=True)
        self._running_speed = LazyProperty(self.api.get_running_speed,
                                           settable=True, lowpass=True)
        self._raw_running_speed = LazyProperty(self.api.get_running_speed,
                                               settable=True, lowpass=False)
        self._stimulus_presentations = LazyProperty(
            self.api.get_stimulus_presentations, settable=True)
        self._stimulus_templates = LazyProperty(
            self.api.get_stimulus_templates, settable=True)
        self._stimulus_timestamps = LazyProperty(
            self.api.get_stimulus_timestamps, settable=True)
        self._task_parameters = LazyProperty(self.api.get_task_parameters,
                                             settable=True)
        self._trials = LazyProperty(self.api.get_trials, settable=True)
        self._metadata = LazyProperty(self.api.get_metadata, settable=True)

    # ==================== class and utility methods ======================

    @classmethod
    def from_lims(cls, behavior_session_id: int) -> "BehaviorSession":
        return cls(api=BehaviorLimsApi(behavior_session_id))

    @classmethod
    def from_nwb_path(
            cls, nwb_path: str, **api_kwargs: Any) -> "BehaviorSession":
        return cls(api=BehaviorNwbApi.from_path(path=nwb_path, **api_kwargs))

    def cache_clear(self) -> None:
        """Convenience method to clear the api cache, if applicable."""
        try:
            self.api.cache_clear()
        except AttributeError:
            logging.getLogger("BehaviorOphysSession").warning(
                "Attempted to clear API cache, but method `cache_clear`"
                f" does not exist on {self.api.__class__.__name__}")

    def list_api_methods(self) -> List[Tuple[str, str]]:
        """Convenience method to expose list of API `get` methods. These
        methods can be accessed by referencing the API used to initialize this
        BehaviorSession via its `api` instance attribute.
        :rtype: list of tuples, where the first value in the tuple is the
        method name, and the second value is the method docstring.
        """
        methods = [m for m in inspect.getmembers(self.api, inspect.ismethod)
                   if m[0].startswith("get_")]
        docs = [inspect.getdoc(m[1]) or "" for m in methods]
        method_names = [m[0] for m in methods]
        return list(zip(method_names, docs))

    # ========================= 'get' methods ==========================

    def get_reward_rate(self):
        response_latency_list = []
        for _, t in self.trials.iterrows():
            valid_response_licks = \
                    [x for x in t.lick_times
                     if x - t.change_time >
                        self.task_parameters['response_window_sec'][0]]
            response_latency = (
                    float('inf')
                    if len(valid_response_licks) == 0
                    else valid_response_licks[0] - t.change_time)
            response_latency_list.append(response_latency)
        reward_rate = calculate_reward_rate(
                response_latency=response_latency_list,
                starttime=self.trials.start_time.values)
        reward_rate[np.isinf(reward_rate)] = float('nan')
        return reward_rate

    def get_rolling_performance_df(self):
        # Indices to build trial metrics dataframe:
        trials_index = self.trials.index
        not_aborted_index = \
            self.trials[np.logical_not(self.trials.aborted)].index

        # Initialize dataframe:
        performance_metrics_df = pd.DataFrame(index=trials_index)

        # Reward rate:
        performance_metrics_df['reward_rate'] = \
            pd.Series(self.get_reward_rate(), index=self.trials.index)

        # Hit rate raw:
        hit_rate_raw = get_hit_rate(
            hit=self.trials.hit,
            miss=self.trials.miss,
            aborted=self.trials.aborted)
        performance_metrics_df['hit_rate_raw'] = \
            pd.Series(hit_rate_raw, index=not_aborted_index)

        # Hit rate with trial count correction:
        hit_rate = get_trial_count_corrected_hit_rate(
                hit=self.trials.hit,
                miss=self.trials.miss,
                aborted=self.trials.aborted)
        performance_metrics_df['hit_rate'] = \
            pd.Series(hit_rate, index=not_aborted_index)

        # False-alarm rate raw:
        false_alarm_rate_raw = \
            get_false_alarm_rate(
                    false_alarm=self.trials.false_alarm,
                    correct_reject=self.trials.correct_reject,
                    aborted=self.trials.aborted)
        performance_metrics_df['false_alarm_rate_raw'] = \
            pd.Series(false_alarm_rate_raw, index=not_aborted_index)

        # False-alarm rate with trial count correction:
        false_alarm_rate = \
            get_trial_count_corrected_false_alarm_rate(
                    false_alarm=self.trials.false_alarm,
                    correct_reject=self.trials.correct_reject,
                    aborted=self.trials.aborted)
        performance_metrics_df['false_alarm_rate'] = \
            pd.Series(false_alarm_rate, index=not_aborted_index)

        # Rolling-dprime:
        rolling_dprime = get_rolling_dprime(hit_rate, false_alarm_rate)
        performance_metrics_df['rolling_dprime'] = \
            pd.Series(rolling_dprime, index=not_aborted_index)

        return performance_metrics_df

    def get_performance_metrics(self, engaged_trial_reward_rate_threshold=2):
        performance_metrics = {}
        performance_metrics['trial_count'] = len(self.trials)
        performance_metrics['go_trial_count'] = self.trials.go.sum()
        performance_metrics['catch_trial_count'] = self.trials.catch.sum()
        performance_metrics['hit_trial_count'] = self.trials.hit.sum()
        performance_metrics['miss_trial_count'] = self.trials.miss.sum()
        performance_metrics['false_alarm_trial_count'] = \
            self.trials.false_alarm.sum()
        performance_metrics['correct_reject_trial_count'] = \
            self.trials.correct_reject.sum()
        performance_metrics['auto_rewarded_trial_count'] = \
            self.trials.auto_rewarded.sum()
        performance_metrics['rewarded_trial_count'] = \
            self.trials.reward_time.apply(lambda x: not np.isnan(x)).sum()
        performance_metrics['total_reward_count'] = len(self.rewards)
        performance_metrics['total_reward_volume'] = self.rewards.volume.sum()

        rpdf = self.get_rpdf()
        engaged_trial_mask = (
                rpdf['reward_rate'] >
                engaged_trial_reward_rate_threshold)
        performance_metrics['maximum_reward_rate'] = \
            np.nanmax(rpdf['reward_rate'].values)
        performance_metrics['engaged_trial_count'] = (engaged_trial_mask).sum()
        performance_metrics['mean_hit_rate'] = \
            rpdf['hit_rate'].mean()
        performance_metrics['mean_hit_rate_uncorrected'] = \
            rpdf['hit_rate_raw'].mean()
        performance_metrics['mean_hit_rate_engaged'] = \
            rpdf['hit_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_false_alarm_rate'] = \
            rpdf['false_alarm_rate'].mean()
        performance_metrics['mean_false_alarm_rate_uncorrected'] = \
            rpdf['false_alarm_rate_raw'].mean()
        performance_metrics['mean_false_alarm_rate_engaged'] = \
            rpdf['false_alarm_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_dprime'] = \
            rpdf['rolling_dprime'].mean()
        performance_metrics['mean_dprime_engaged'] = \
            rpdf['rolling_dprime'][engaged_trial_mask].mean()
        performance_metrics['max_dprime'] = \
            rpdf['rolling_dprime'].max()
        performance_metrics['max_dprime_engaged'] = \
            rpdf['rolling_dprime'][engaged_trial_mask].max()

        return performance_metrics

    # ====================== properties and setters ========================

    @property
    def behavior_session_id(self) -> int:
        """Unique identifier for this experimental session.
        :rtype: int
        """
        return self._behavior_session_id

    @property
    def licks(self) -> pd.DataFrame:
        """Get lick data from pkl file.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for BehaviorOphysSessions.

        Returns
        -------
        np.ndarray
            A dataframe containing lick timestamps.
        """
        return self._licks

    @licks.setter
    def licks(self, value):
        self._licks = value

    @property
    def rewards(self) -> pd.DataFrame:
        """Get reward data from pkl file.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for BehaviorOphysSessions.

        Returns
        -------
        pd.DataFrame
            A dataframe containing timestamps of delivered rewards.
        """
        return self._rewards

    @rewards.setter
    def rewards(self, value):
        self._rewards = value

    @property
    def running_speed(self) -> pd.DataFrame:
        """Get running speed data. By default applies a 10Hz low pass
        filter to the data. To get the running speed without the filter,
        use `raw_running_speed`.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for BehaviorOphysSessions.

        Returns
        -------
        pd.DataFrame
            Dataframe containing various signals used to compute running
            speed, and the filtered speed.
        """
        return self._running_speed

    @running_speed.setter
    def running_speed(self, value):
        self._running_speed = value

    @property
    def raw_running_speed(self) -> pd.DataFrame:
        """Get unfiltered running speed data.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for BehaviorOphysSessions.

        Returns
        -------
        pd.DataFrame
            Dataframe containing various signals used to compute running
            speed, and the unfiltered speed.
        """
        return self._raw_running_speed

    @raw_running_speed.setter
    def raw_running_speed(self, value):
        self._raw_running_speed = value

    @property
    def stimulus_presentations(self) -> pd.DataFrame:
        """Table whose rows are stimulus presentations (i.e. a given image,
        for a given duration, typically 250 ms) and whose columns are
        presentation characteristics.

        Returns
        -------
        pd.DataFrame
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        return self._stimulus_presentations

    @stimulus_presentations.setter
    def stimulus_presentations(self, value):
        self._stimulus_presentations = value

    @property
    def stimulus_templates(self) -> StimulusTemplate:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        StimulusTemplate
            A StimulusTemplate object containing the stimulus images for the
            experiment. Relevant properties include:
                image_set_name: The name of the image set that the
                    StimulusTemplate encapsulates
                image_names: A list of individual image names in the image set
                images: A list of StimulusImage (inherits from np.ndarray)
                    objects.
            Also has a to_dataframe() method to convert to a dataframe
            where indices are image names, an 'image' column contains image
            arrays, and the df.name is the image set.
        """
        return self._stimulus_templates

    @stimulus_templates.setter
    def stimulus_templates(self, value):
        self._stimulus_templates = value

    @property
    def stimulus_timestamps(self) -> np.ndarray:
        """Get stimulus timestamps from pkl file.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for BehaviorOphysSessions.

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
        """
        return self._stimulus_timestamps

    @stimulus_timestamps.setter
    def stimulus_timestamps(self, value):
        self._stimulus_timestamps = value

    @property
    def task_parameters(self) -> dict:
        """Get task parameters from pkl file.

        Returns
        -------
        dict
            A dictionary containing parameters used to define the task runtime
            behavior.
        """
        return self._task_parameters

    @task_parameters.setter
    def task_parameters(self, value):
        self._task_parameters = value

    @property
    def trials(self) -> pd.DataFrame:
        """Get trials from pkl file

        Returns
        -------
        pd.DataFrame
            A dataframe containing behavioral trial start/stop times,
            and trial data
        """
        return self._trials

    @trials.setter
    def trials(self, value):
        self._trials = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the session.
        :rtype: dict
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
