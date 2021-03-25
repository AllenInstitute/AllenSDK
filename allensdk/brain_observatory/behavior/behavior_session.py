from typing import Any, Optional, List, Dict, Type, Tuple
import logging
import pandas as pd
import numpy as np
import inspect

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.core.lazy_property import LazyPropertyMixin
from allensdk.brain_observatory.session_api_utils import ParamsMixin
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsApi, BehaviorNwbApi)
from allensdk.brain_observatory.behavior.session_apis.abcs.\
    session_base.behavior_base import BehaviorBase
from allensdk.brain_observatory.behavior.trials_processing import (
    construct_rolling_performance_df, calculate_reward_rate_fix_nans)


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
            logging.getLogger("BehaviorSession").warning(
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

    def list_data_attributes_and_methods(self) -> List[str]:
        """Convenience method for end-users to list attributes and methods
        that can be called to access data for a BehaviorSession.

        NOTE: Because BehaviorOphysExperiment inherits from BehaviorSession,
        this method will also be available there.

        Returns
        -------
        List[str]
            A list of attributes and methods that end-users can access or call
            to get data.
        """
        attrs_and_methods_to_ignore: set = {
            "api",
            "cache_clear",
            "from_lims",
            "from_nwb_path",
            "LazyProperty",
            "list_api_methods",
            "list_data_attributes_and_methods"
        }
        attrs_and_methods_to_ignore.update(dir(ParamsMixin))
        attrs_and_methods_to_ignore.update(dir(LazyPropertyMixin))
        class_dir = dir(self)
        attrs_and_methods = [
            r for r in class_dir
            if (r not in attrs_and_methods_to_ignore and not r.startswith("_"))
        ]
        return attrs_and_methods

    # ========================= 'get' methods ==========================

    def get_reward_rate(self) -> np.ndarray:
        """ Get the reward rate of the subject for the task calculated over a
        25 trial rolling window and provides a measure of the rewards
        earned per unit time (in units of rewards/minute).

        Returns
        -------
        np.ndarray
            The reward rate (rewards/minute) of the subject for the
            task calculated over a 25 trial rolling window.
        """
        return calculate_reward_rate_fix_nans(
                self.trials,
                self.task_parameters['response_window_sec'][0])

    def get_rolling_performance_df(self) -> pd.DataFrame:
        """Return a DataFrame containing trial by trial behavior response
        performance metrics.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing:
                trials_id [index]:
                    Index of the trial. All trials, including aborted trials,
                    are assigned an index starting at 0 for the first trial.
                reward_rate:
                    Rewards earned in the previous 25 trials, normalized by
                    the elapsed time of the same 25 trials. Units are
                    rewards/minute.
                hit_rate_raw:
                    Fraction of go trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                hit_rate:
                    Fraction of go trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. With trial count correction applied.
                false_alarm_rate_raw:
                    Fraction of catch trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                false_alarm_rate:
                    Fraction of catch trials where the mouse licked in
                    the response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                rolling_dprime:
                    d prime calculated using the rolling hit_rate and
                    rolling false_alarm _rate.

        """
        return construct_rolling_performance_df(
                self.trials,
                self.task_parameters['response_window_sec'][0],
                self.task_parameters["session_type"])

    def get_performance_metrics(
            self,
            engaged_trial_reward_rate_threshold: float = 2.0
            ) -> dict:
        """Get a dictionary containing a subject's behavior response
        summary data.

        Parameters
        ----------
        engaged_trial_reward_rate_threshold : float, optional
            The number of rewards per minute that needs to be attained
            before a subject is considered 'engaged', by default 2.0

        Returns
        -------
        dict
            Returns a dict of performance metrics with the following fields:
                trial_count:
                    The length of the trial dataframe
                    (including all 'go', 'catch', and 'aborted' trials)
                go_trial_count:
                    Number of 'go' trials in a behavior session
                catch_trial_count:
                    Number of 'catch' trial types during a behavior session
                hit_trial_count:
                    Number of trials with a hit behavior response
                    type in a behavior session
                miss_trial_count:
                    Number of trials with a miss behavior response
                    type in a behavior session
                false_alarm_trial_count:
                    Number of trials where the mouse had a false alarm
                    behavior response
                correct_reject_trial_count:
                    Number of trials with a correct reject behavior
                    response during a behavior session
                auto_rewarded_trial_count:
                    Number of trials where the mouse received an auto
                    reward of water.
                rewarded_trial_count:
                    Number of trials on which the animal was rewarded for
                    licking in the response window.
                total_reward_count:
                    Number of trials where the mouse received a
                    water reward (earned or auto rewarded)
                total_reward_volume:
                    Volume of all water rewards received during a
                    behavior session (earned and auto rewarded)
                maximum_reward_rate:
                    The peak of the rolling reward rate (rewards/minute)
                engaged_trial_count:
                    Number of trials where the mouse is engaged
                    (reward rate > 2 rewards/minute)
                mean_hit_rate:
                    The mean of the rolling hit_rate
                mean_hit_rate_uncorrected:
                    The mean of the rolling hit_rate_raw
                mean_hit_rate_engaged:
                    The mean of the rolling hit_rate, excluding epochs
                    when the rolling reward rate was below 2 rewards/minute
                mean_false_alarm_rate:
                    The mean of the rolling false_alarm_rate, excluding
                    epochs when the rolling reward rate was below 2
                    rewards/minute
                mean_false_alarm_rate_uncorrected:
                    The mean of the rolling false_alarm_rate_raw
                mean_false_alarm_rate_engaged:
                    The mean of the rolling false_alarm_rate,
                    excluding epochs when the rolling reward rate
                    was below 2 rewards/minute
                mean_dprime:
                    The mean of the rolling d_prime
                mean_dprime_engaged:
                    The mean of the rolling d_prime, excluding
                    epochs when the rolling reward rate was
                    below 2 rewards/minute
                max_dprime:
                    The peak of the rolling d_prime
                max_dprime_engaged:
                    The peak of the rolling d_prime, excluding epochs
                    when the rolling reward rate was below 2 rewards/minute
        """
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
        # Although 'rewarded_trial_count' will currently have the same value as
        # 'hit_trial_count', in the future there may be variants of the
        # task where rewards are withheld. In that case the
        # 'rewarded_trial_count' will be smaller than (and different from)
        # the 'hit_trial_count'.
        performance_metrics['rewarded_trial_count'] = self.trials.hit.sum()
        performance_metrics['total_reward_count'] = len(self.rewards)
        performance_metrics['total_reward_volume'] = self.rewards.volume.sum()

        rpdf = self.get_rolling_performance_df()
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
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

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
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

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
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

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
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

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
    def stimulus_templates(self) -> pd.DataFrame:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame object containing the stimulus images for the
            experiment. Indices are image names, 'warped' and 'unwarped'
            columns contains image arrays, and the df.name is the image set.
        """
        return self._stimulus_templates.to_dataframe()

    @stimulus_templates.setter
    def stimulus_templates(self, value):
        self._stimulus_templates = value

    @property
    def stimulus_timestamps(self) -> np.ndarray:
        """Get stimulus timestamps from pkl file.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

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
                auto_reward_volume: (float)
                    Volume of auto rewards in ml.
                blank_duration_sec : (list of floats)
                    Duration in seconds of inter stimulus interval.
                    Inter-stimulus interval chosen as a uniform random value.
                    between the range defined by the two values.
                    Values are ignored if `stimulus_duration_sec` is null.
                response_window_sec: (list of floats)
                    Range of period following an image change, in seconds,
                    where mouse response influences trial outcome.
                    First value represents response window start.
                    Second value represents response window end.
                    Values represent time before display lag is
                    accounted for and applied.
                n_stimulus_frames: (int)
                    Total number of visual stimulus frames presented during
                    a behavior session.
                task: (string)
                    Type of visual stimulus task.
                session_type: (string)
                    Visual stimulus type run during behavior session.
                omitted_flash_fraction: (float)
                    Probability that a stimulus image presentations is omitted.
                    Change stimuli, and the stimulus immediately preceding the
                    change, are never omitted.
                stimulus_distribution: (string)
                    Distribution for drawing change times.
                    Either 'exponential' or 'geometric'.
                stimulus_duration_sec: (float)
                    Duration in seconds of each stimulus image presentation
                reward_volume: (float)
                    Volume of earned water reward in ml.
                stimulus: (string)
                    Stimulus type ('gratings' or 'images').

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
        if isinstance(self._metadata, BehaviorMetadata):
            metadata = self._metadata.to_dict()
        else:
            # NWB API returns as dict
            metadata = self._metadata

        return metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
