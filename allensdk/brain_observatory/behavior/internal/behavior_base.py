import abc

from typing import Dict, NamedTuple

import numpy as np
import pandas as pd
from allensdk.brain_observatory.running_speed import RunningSpeed


class BehaviorBase(abc.ABC):
    """Abstract base class implementing required methods for interacting with
    behavior session data.

    Child classes should be instantiated with a fetch API that implements these
    methods.
    """
    @abc.abstractmethod
    def get_licks(self) -> pd.DataFrame:
        """Get lick data from pkl file.

        Returns
        -------
        np.ndarray
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
    def get_running_data_df(self) -> pd.DataFrame:
        """Get running speed data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing various signals used to compute running speed.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_running_speed(self) -> RunningSpeed:
        """Get running speed using timestamps from
        self.get_stimulus_timestamps.

        NOTE: Do not correct for monitor delay.

        Returns
        -------
        RunningSpeed (NamedTuple with two fields)
            timestamps : np.ndarray
                Timestamps of running speed data samples
            values : np.ndarray
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
    def get_stimulus_templates(self) -> Dict[str, np.ndarray]:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the stimulus images presented during the
            session. Keys are data set names, and values are 3D numpy arrays.
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
