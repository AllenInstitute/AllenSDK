import abc
from typing import Union

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.stimulus_processing import \
    StimulusTemplate


class BehaviorBase(abc.ABC):
    """Abstract base class implementing required methods for interacting with
    behavior session data.

    Child classes should be instantiated with a fetch API that implements these
    methods.
    """
    @abc.abstractmethod
    def get_behavior_session_id(self) -> int:
        """Returns the behavior_session_id associated with this experiment,
        if applicable.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_licks(self) -> pd.DataFrame:
        """Get lick data from pkl file.

        Returns
        -------
        pd.Dataframe
            A dataframe containing lick timestamps.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rewards(self) -> pd.DataFrame:
        """Get reward data from pkl file.

        Returns
        -------
        pd.DataFrame
            A dataframe containing timestamps of delivered rewards.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_running_acquisition_df(self) -> pd.DataFrame:
        """Get running speed acquisition data from a behavior pickle file.

        Returns
        -------
        pd.DataFrame
            Dataframe with an index of timestamps and the following columns:
                "speed": computed running speed
                "dx": angular change, computed during data collection
                "v_sig": voltage signal from the encoder
                "v_in": the theoretical maximum voltage that the encoder
                    will reach prior to "wrapping". This should
                    theoretically be 5V (after crossing 5V goes to 0V, or
                    vice versa). In practice the encoder does not always
                    reach this value before wrapping, which can cause
                    transient spikes in speed at the voltage "wraps".
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_running_speed(self) -> pd.DataFrame:
        """Get running speed using timestamps from
        self.get_stimulus_timestamps.

        NOTE: Do not correct for monitor delay.

        Returns
        -------
        pd.DataFrame
            timestamps : np.ndarray
                index consisting of timestamps of running speed data samples
            speed : np.ndarray
                Running speed of the experimental subject (in cm / s).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stimulus_presentations(self) -> pd.DataFrame:
        """Get stimulus presentation data.

        NOTE: Uses timestamps that do not account for monitor delay.

        Returns
        -------
        pd.DataFrame
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stimulus_templates(self) -> StimulusTemplate:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        StimulusTemplate
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stimulus_timestamps(self) -> np.ndarray:
        """Get stimulus timestamps from pkl file.

        NOTE: Located with behavior_session_id

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
            that do no account for monitor delay.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_task_parameters(self) -> dict:
        """Get task parameters from pkl file.

        Returns
        -------
        dict
            A dictionary containing parameters used to define the task runtime
            behavior.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_trials(self) -> pd.DataFrame:
        """Get trials from pkl file

        Returns
        -------
        pd.DataFrame
            A dataframe containing behavioral trial start/stop times,
            and trial data
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_metadata(self) -> Union[BehaviorMetadata, dict]:
        """Get metadata for Session

        Returns
        -------
        dict if NWB
        BehaviorMetadata otherwise
        """
        raise NotImplementedError()
