import datetime
from typing import Any, List, Dict
import pynwb
import pandas as pd
import numpy as np
import pytz

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    InternalReadableInterface, JsonReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.licks import Licks
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata, get_expt_description
from allensdk.brain_observatory.behavior.data_objects.rewards import Rewards
from allensdk.brain_observatory.behavior.data_objects.stimuli.stimuli import \
    Stimuli
from allensdk.brain_observatory.behavior.data_objects.task_parameters import \
    TaskParameters
from allensdk.brain_observatory.behavior.data_objects.trials.trial_table \
    import \
    TrialTable
from allensdk.brain_observatory.behavior.trials_processing import (
    construct_rolling_performance_df, calculate_reward_rate_fix_nans)
from allensdk.brain_observatory.behavior.data_objects import (
    BehaviorSessionId, StimulusTimestamps, RunningSpeed, RunningAcquisition,
    DataObject
)

from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator


class BehaviorSession(DataObject, InternalReadableInterface,
                      NwbReadableInterface,
                      JsonReadableInterface, NwbWritableInterface):
    def __init__(
        self,
        behavior_session_id: BehaviorSessionId,
        stimulus_timestamps: StimulusTimestamps,
        running_acquisition: RunningAcquisition,
        raw_running_speed: RunningSpeed,
        running_speed: RunningSpeed,
        licks: Licks,
        rewards: Rewards,
        stimuli: Stimuli,
        task_parameters: TaskParameters,
        trials: TrialTable,
        metadata: BehaviorMetadata
    ):
        super().__init__(name='behavior_session', value=self)

        self._behavior_session_id = behavior_session_id
        self._licks = licks
        self._rewards = rewards
        self._running_acquisition = running_acquisition
        self._running_speed = running_speed
        self._raw_running_speed = raw_running_speed
        self._stimuli = stimuli
        self._stimulus_timestamps = stimulus_timestamps
        self._task_parameters = task_parameters
        self._trials = trials
        self._metadata = metadata

    # ==================== class and utility methods ======================

    @classmethod
    def from_json(cls, session_data: dict) -> "BehaviorSession":
        behavior_session_id = BehaviorSessionId.from_json(
            dict_repr=session_data)
        stimulus_timestamps = StimulusTimestamps.from_json(
            dict_repr=session_data)
        running_acquisition = RunningAcquisition.from_json(
            dict_repr=session_data)
        raw_running_speed = RunningSpeed.from_json(
            dict_repr=session_data, filtered=False
        )
        running_speed = RunningSpeed.from_json(dict_repr=session_data)
        metadata = BehaviorMetadata.from_json(dict_repr=session_data)

        stimulus_file = StimulusFile.from_json(dict_repr=session_data)
        licks, rewards, stimuli, task_parameters, trials = \
            cls._read_data_from_stimulus_file(
                stimulus_file=stimulus_file,
                stimulus_timestamps=stimulus_timestamps
            )

        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            stimulus_timestamps=stimulus_timestamps,
            running_acquisition=running_acquisition,
            raw_running_speed=raw_running_speed,
            running_speed=running_speed,
            metadata=metadata,
            licks=licks,
            rewards=rewards,
            stimuli=stimuli,
            task_parameters=task_parameters,
            trials=trials
        )

    @classmethod
    def from_internal(cls, behavior_session_id: int) -> "BehaviorSession":
        lims_db = db_connection_creator(
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP
        )

        behavior_session_id = BehaviorSessionId(behavior_session_id)
        stimulus_file = StimulusFile.from_lims(
            db=lims_db, behavior_session_id=behavior_session_id.value)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file
        )
        running_acquisition = RunningAcquisition.from_lims(
            lims_db, behavior_session_id.value
        )
        raw_running_speed = RunningSpeed.from_lims(
            lims_db, behavior_session_id.value, filtered=False
        )
        running_speed = RunningSpeed.from_lims(
            lims_db, behavior_session_id.value
        )
        behavior_metadata = BehaviorMetadata.from_internal(
            behavior_session_id=behavior_session_id, lims_db=lims_db
        )
        stimulus_file = StimulusFile.from_lims(
            db=lims_db, behavior_session_id=behavior_session_id.value)
        licks, rewards, stimuli, task_parameters, trials = \
            cls._read_data_from_stimulus_file(
                stimulus_file=stimulus_file,
                stimulus_timestamps=stimulus_timestamps
            )

        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            stimulus_timestamps=stimulus_timestamps,
            running_acquisition=running_acquisition,
            raw_running_speed=raw_running_speed,
            running_speed=running_speed,
            metadata=behavior_metadata,
            licks=licks,
            rewards=rewards,
            stimuli=stimuli,
            task_parameters=task_parameters,
            trials=trials
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "BehaviorSession":
        behavior_session_id = BehaviorSessionId.from_nwb(nwbfile)
        stimulus_timestamps = StimulusTimestamps.from_nwb(nwbfile)
        running_acquisition = RunningAcquisition.from_nwb(nwbfile)
        raw_running_speed = RunningSpeed.from_nwb(nwbfile, filtered=False)
        running_speed = RunningSpeed.from_nwb(nwbfile)
        metadata = BehaviorMetadata.from_nwb(nwbfile)
        licks = Licks.from_nwb(nwbfile=nwbfile)
        rewards = Rewards.from_nwb(nwbfile=nwbfile)
        stimuli = Stimuli.from_nwb(nwbfile=nwbfile)
        task_parameters = TaskParameters.from_nwb(nwbfile=nwbfile)
        trials = TrialTable.from_nwb(nwbfile=nwbfile)

        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            stimulus_timestamps=stimulus_timestamps,
            running_acquisition=running_acquisition,
            raw_running_speed=raw_running_speed,
            running_speed=running_speed,
            metadata=metadata,
            licks=licks,
            rewards=rewards,
            stimuli=stimuli,
            task_parameters=task_parameters,
            trials=trials
        )

    @classmethod
    def from_nwb_path(cls, nwb_path: str):
        with pynwb.NWBHDF5IO(str(nwb_path), 'r') as read_io:
            nwbfile = read_io.read()
            return cls.from_nwb(nwbfile=nwbfile)

    def to_nwb(self) -> NWBFile:
        nwbfile = NWBFile(
            session_description=self._metadata.session_type,
            identifier=str(self.behavior_session_id),
            session_start_time=self._metadata.date_of_acquisition,
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=["visual", "behavior", "task"],
            experiment_description=get_expt_description(
                session_type=self._metadata.session_type)
        )

        self._stimulus_timestamps.to_nwb(nwbfile=nwbfile)
        self._running_acquisition.to_nwb(nwbfile=nwbfile)
        self._raw_running_speed.to_nwb(nwbfile=nwbfile)
        self._running_speed.to_nwb(nwbfile=nwbfile)
        self._metadata.to_nwb(nwbfile=nwbfile)
        self._licks.to_nwb(nwbfile=nwbfile)
        self._rewards.to_nwb(nwbfile=nwbfile)
        self._stimuli.to_nwb(nwbfile=nwbfile)
        self._task_parameters.to_nwb(nwbfile=nwbfile)
        self._trials.to_nwb(nwbfile=nwbfile)

        return nwbfile

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
            "from_json",
            "from_lims",
            "from_nwb_path",
            "list_data_attributes_and_methods"
        }
        attrs_and_methods_to_ignore.update(dir(InternalReadableInterface))
        attrs_and_methods_to_ignore.update(dir(NwbReadableInterface))
        attrs_and_methods_to_ignore.update(dir(NwbWritableInterface))
        attrs_and_methods_to_ignore.update(dir(DataObject))
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
                trials_id [index]: (int)
                    Index of the trial. All trials, including aborted trials,
                    are assigned an index starting at 0 for the first trial.
                reward_rate: (float)
                    Rewards earned in the previous 25 trials, normalized by
                    the elapsed time of the same 25 trials. Units are
                    rewards/minute.
                hit_rate_raw: (float)
                    Fraction of go trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                hit_rate: (float)
                    Fraction of go trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. With trial count correction applied.
                false_alarm_rate_raw: (float)
                    Fraction of catch trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                false_alarm_rate: (float)
                    Fraction of catch trials where the mouse licked in
                    the response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                rolling_dprime: (float)
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
                trial_count: (int)
                    The length of the trial dataframe
                    (including all 'go', 'catch', and 'aborted' trials)
                go_trial_count: (int)
                    Number of 'go' trials in a behavior session
                catch_trial_count: (int)
                    Number of 'catch' trial types during a behavior session
                hit_trial_count: (int)
                    Number of trials with a hit behavior response
                    type in a behavior session
                miss_trial_count: (int)
                    Number of trials with a miss behavior response
                    type in a behavior session
                false_alarm_trial_count: (int)
                    Number of trials where the mouse had a false alarm
                    behavior response
                correct_reject_trial_count: (int)
                    Number of trials with a correct reject behavior
                    response during a behavior session
                auto_reward_count:
                    Number of trials where the mouse received an auto
                    reward of water.
                earned_reward_count:
                    Number of trials where the mouse was eligible to receive a
                    water reward ('go' trials) and did receive an earned
                    water reward
                total_reward_count:
                    Number of trials where the mouse received a
                    water reward (earned or auto rewarded)
                total_reward_volume: (float)
                    Volume of all water rewards received during a
                    behavior session (earned and auto rewarded)
                maximum_reward_rate: (float)
                    The peak of the rolling reward rate (rewards/minute)
                engaged_trial_count: (int)
                    Number of trials where the mouse is engaged
                    (reward rate > 2 rewards/minute)
                mean_hit_rate: (float)
                    The mean of the rolling hit_rate
                mean_hit_rate_uncorrected:
                    The mean of the rolling hit_rate_raw
                mean_hit_rate_engaged: (float)
                    The mean of the rolling hit_rate, excluding epochs
                    when the rolling reward rate was below 2 rewards/minute
                mean_false_alarm_rate: (float)
                    The mean of the rolling false_alarm_rate, excluding
                    epochs when the rolling reward rate was below 2
                    rewards/minute
                mean_false_alarm_rate_uncorrected: (float)
                    The mean of the rolling false_alarm_rate_raw
                mean_false_alarm_rate_engaged: (float)
                    The mean of the rolling false_alarm_rate,
                    excluding epochs when the rolling reward rate
                    was below 2 rewards/minute
                mean_dprime: (float)
                    The mean of the rolling d_prime
                mean_dprime_engaged: (float)
                    The mean of the rolling d_prime, excluding
                    epochs when the rolling reward rate was
                    below 2 rewards/minute
                max_dprime: (float)
                    The peak of the rolling d_prime
                max_dprime_engaged: (float)
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
        performance_metrics['auto_reward_count'] = \
            self.trials.auto_rewarded.sum()
        # Although 'earned_reward_count' will currently have the same value as
        # 'hit_trial_count', in the future there may be variants of the
        # task where rewards are withheld. In that case the
        # 'earned_reward_count' will be smaller than (and different from)
        # the 'hit_trial_count'.
        performance_metrics['earned_reward_count'] = self.trials.hit.sum()
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
        """Unique identifier for a behavioral session.
        :rtype: int
        """
        return self._behavior_session_id.value

    @property
    def licks(self) -> pd.DataFrame:
        """A dataframe containing lick timestmaps and frames, sampled
        at 60 Hz.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        np.ndarray
            A dataframe containing lick timestamps.
            dataframe columns:
                timestamps: (float)
                    time of lick, in seconds
                frame: (int)
                    frame of lick

        """
        return self._licks.value

    @licks.setter
    def licks(self, value):
        self._licks = value

    @property
    def rewards(self) -> pd.DataFrame:
        """Retrieves rewards from data file saved at the end of the
        behavior session.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        pd.DataFrame
            A dataframe containing timestamps of delivered rewards.
            Timestamps are sampled at 60Hz.

            dataframe columns:
                volume: (float)
                    volume of individual water reward in ml.
                    0.007 if earned reward, 0.005 if auto reward.
                timestamps: (float)
                    time in seconds
                autorewarded: (bool)
                    True if free reward was delivered for that trial.
                    Occurs during the first 5 trials of a session and
                    throughout as needed

        """
        return self._rewards.value

    @rewards.setter
    def rewards(self, value):
        self._rewards = value

    @property
    def running_speed(self) -> pd.DataFrame:
        """Running speed and timestamps, sampled at 60Hz. By default
        applies a 10Hz low pass filter to the data. To get the
        running speed without the filter, use `raw_running_speed`.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        pd.DataFrame
            Dataframe containing running speed and timestamps
            dataframe columns:
                timestamps: (float)
                    time in seconds
                speed: (float)
                    speed in cm/sec
        """
        return self._running_speed.value

    @running_speed.setter
    def running_speed(self, value):
        self._running_speed.value = value

    @property
    def raw_running_speed(self) -> pd.DataFrame:
        """Get unfiltered running speed data. Sampled at 60Hz.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        pd.DataFrame
            Dataframe containing unfiltered running speed and timestamps
            dataframe columns:
                timestamps: (float)
                    time in seconds
                speed: (float)
                    speed in cm/sec
        """
        return self._raw_running_speed.value

    @raw_running_speed.setter
    def raw_running_speed(self, value):
        self._raw_running_speed.value = value

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

            dataframe columns:
                stimulus_presentations_id [index]: (int)
                    identifier for a stimulus presentation
                    (presentation of an image)
                duration: (float)
                    duration of an image presentation (flash)
                    in seconds (stop_time - start_time). NaN if omitted
                end_frame: (float)
                    image presentation end frame
                image_index: (int)
                    image index (0-7) for a given session,
                    corresponding to each image name
                image_set: (string)
                    image set for this behavior session
                index: (int)
                    an index assigned to each stimulus presentation
                omitted: (bool)
                    True if no image was shown for this stimulus
                    presentation
                start_frame: (int)
                    image presentation start frame
                start_time: (float)
                    image presentation start time in seconds
                stop_time: (float)
                    image presentation end time in seconds
        """
        return self._stimuli.presentations.value

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
            experiment.

            dataframe columns:
                image_name [index]: (string)
                    name of image presented, if 'omitted'
                    then no image was presented
                unwarped: (array of int)
                    image array of unwarped stimulus image
                warped: (array of int)
                    image array of warped stimulus image

        """
        return self._stimuli.templates.value.to_dataframe()

    @stimulus_templates.setter
    def stimulus_templates(self, value):
        self._stimulus_templates = value

    @property
    def stimulus_timestamps(self) -> np.ndarray:
        """Timestamps associated with the stimulus presetntation on
        the monitor retrieveddata file saved at the end of the
        behavior session. Sampled at 60Hz.

        NOTE: For BehaviorSessions, returned timestamps are not
        aligned to external 'synchronization' reference timestamps.
        Synchronized timestamps are only available for
        BehaviorOphysExperiments.

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
        """
        return self._stimulus_timestamps.value

    @stimulus_timestamps.setter
    def stimulus_timestamps(self, value):
        self._stimulus_timestamps.value = value

    @property
    def task_parameters(self) -> dict:
        """Get task parameters from data file saved at the end of
        the behavior session file.

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
        return self._task_parameters.to_dict()['task_parameters']

    @task_parameters.setter
    def task_parameters(self, value):
        self._task_parameters = value

    @property
    def trials(self) -> pd.DataFrame:
        """Get trials from data file saved at the end of the
        behavior session.

        Returns
        -------
        pd.DataFrame
            A dataframe containing trial and behavioral response data,
            by cell specimen id

            dataframe columns:
                trials_id: (int)
                    trial identifier
                lick_times: (array of float)
                    array of lick times in seconds during that trial.
                    Empty array if no licks occured during the trial.
                reward_time: (NaN or float)
                    Time the reward is delivered following a correct
                    response or on auto rewarded trials.
                reward_volume: (float)
                    volume of reward in ml. 0.005 for auto reward
                    0.007 for earned reward
                hit: (bool)
                    Behavior response type. On catch trial mouse licks
                    within reward window.
                false_alarm: (bool)
                    Behavior response type. On catch trial mouse licks
                    within reward window.
                miss: (bool)
                    Behavior response type. On a go trial, mouse either
                    does not lick at all, or licks after reward window
                stimulus_change: (bool)
                    True if an image change occurs during the trial
                    (if the trial was both a 'go' trial and the trial
                    was not aborted)
                aborted: (bool)
                    Behavior response type. True if the mouse licks
                    before the scheduled change time.
                go: (bool)
                    Trial type. True if there was a change in stimulus
                    image identity on this trial
                catch: (bool)
                    Trial type. True if there was not a change in stimulus
                    identity on this trial
                auto_rewarded: (bool)
                    True if free reward was delivered for that trial.
                    Occurs during the first 5 trials of a session and
                    throughout as needed.
                correct_reject: (bool)
                    Behavior response type. On a catch trial, mouse
                    either does not lick at all or licks after reward
                    window
                start_time: (float)
                    start time of the trial in seconds
                stop_time: (float)
                    end time of the trial in seconds
                trial_length: (float)
                    duration of trial in seconds (stop_time -start_time)
                response_time: (float)
                    time of first lick in trial in seconds and NaN if
                    trial aborted
                initial_image_name: (string)
                    name of image presented at start of trial
                change_image_name: (string)
                    name of image that is changed to at the change time,
                    on go trials
        """
        return self._trials.value

    @trials.setter
    def trials(self, value):
        self._trials = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """metadata for a give session

        Returns
        -------
        Dict
            A dictionary containing behavior session specific metadata
            dictionary keys:
                age_in_days: (int)
                    age of mouse in days
                behavior_session_uuid: (int)
                    unique identifier for a behavior session
                behavior_session_id: (int)
                    unique identifier for a behavior session
                cre_line: (string)
                    cre driver line for a transgenic mouse
                date_of_acquisition: (date time object)
                    date and time of experiment acquisition,
                    yyyy-mm-dd hh:mm:ss
                driver_line: (list of string)
                    all driver lines for a transgenic mouse
                equipment_name: (string)
                    identifier for equipment data was collected on
                full_genotype: (string)
                    full genotype of transgenic mouse
                mouse_id: (int)
                    unique identifier for a mouse
                reporter_line: (string)
                    reporter line for a transgenic mouse
                session_type: (string)
                    visual stimulus type displayed during behavior
                    session
                sex: (string)
                    sex of the mouse
                stimulus_frame_rate: (float)
                    frame rate (Hz) at which the visual stimulus is
                    displayed
        """
        return {
            'equipment_name': self._metadata.equipment.value,
            'sex': self._metadata.subject_metadata.sex,
            'age_in_days': self._metadata.subject_metadata.age_in_days,
            'stimulus_frame_rate': self._metadata.stimulus_frame_rate,
            'session_type': self._metadata.session_type,
            'date_of_acquisition': self._metadata.date_of_acquisition,
            'reporter_line': self._metadata.subject_metadata.reporter_line,
            'cre_line': self._metadata.subject_metadata.cre_line,
            'behavior_session_uuid': self._metadata.behavior_session_uuid,
            'driver_line': self._metadata.subject_metadata.driver_line,
            'mouse_id': self._metadata.subject_metadata.mouse_id,
            'full_genotype': self._metadata.subject_metadata.full_genotype,
            'behavior_session_id': self._metadata.behavior_session_id
        }

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @staticmethod
    def _read_data_from_stimulus_file(stimulus_file: StimulusFile,
                                      stimulus_timestamps: StimulusTimestamps):
        """Helper method to read data from stimulus file"""
        licks = Licks.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        rewards = Rewards.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        stimuli = Stimuli.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        task_parameters = TaskParameters.from_stimulus_file(
            stimulus_file=stimulus_file)
        trials = TrialTable.from_stimulus_file(
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            licks=licks,
            rewards=rewards,
            # This is the median estimate across all rigs
            # as discussed in
            # https://github.com/AllenInstitute/AllenSDK/issues/1318
            monitor_delay=0.02115
        )
        return licks, rewards, stimuli, task_parameters, trials
