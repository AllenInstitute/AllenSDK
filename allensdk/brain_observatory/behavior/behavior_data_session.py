from typing import Any, Optional, List, Dict, Type, Tuple
import logging
import pandas as pd
import numpy as np
import inspect

from allensdk.internal.api.behavior_data_lims_api import BehaviorDataLimsApi
from allensdk.brain_observatory.behavior.internal import BehaviorBase
from allensdk.brain_observatory.running_speed import RunningSpeed

BehaviorDataApi = Type[BehaviorBase]


class BehaviorDataSession(object):
    def __init__(self, api: Optional[BehaviorDataApi] = None):
        self.api = api

    @classmethod
    def from_lims(cls, behavior_session_id: int) -> "BehaviorDataSession":
        return cls(api=BehaviorDataLimsApi(behavior_session_id))

    @classmethod
    def from_nwb_path(
            cls, nwb_path: str, **api_kwargs: Any) -> "BehaviorDataSession":
        return NotImplementedError

    @property
    def behavior_session_id(self) -> int:
        """Unique identifier for this experimental session.
        :rtype: int
        """
        return self.api.behavior_session_id

    @property
    def ophys_session_id(self) -> Optional[int]:
        """The unique identifier for the ophys session associated
        with this behavior session (if one exists)
        :rtype: int
        """
        return self.api.ophys_session_id

    @property
    def ophys_experiment_ids(self) -> Optional[List[int]]:
        """The unique identifiers for the ophys experiment(s) associated
        with this behavior session (if one exists)
        :rtype: int
        """
        return self.api.ophys_experiment_ids

    @property
    def licks(self) -> pd.DataFrame:
        """Get lick data from pkl file.

        Returns
        -------
        np.ndarray
            A dataframe containing lick timestamps.
        """
        return self.api.get_licks()

    @property
    def rewards(self) -> pd.DataFrame:
        """Get reward data from pkl file.

        Returns
        -------
        pd.DataFrame
            A dataframe containing timestamps of delivered rewards.
        """
        return self.api.get_rewards()

    @property
    def running_data_df(self) -> pd.DataFrame:
        """Get running speed data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing various signals used to compute running speed.
        """
        return self.api.get_running_data_df()

    @property
    def running_speed(self) -> RunningSpeed:
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
        return self.api.get_running_speed()

    @property
    def stimulus_presentations(self) -> pd.DataFrame:
        """Get stimulus presentation data.

        NOTE: Uses timestamps that do not account for monitor delay.

        Returns
        -------
        pd.DataFrame
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        return self.api.get_stimulus_presentations()

    @property
    def stimulus_templates(self) -> Dict[str, np.ndarray]:
        """Get stimulus templates (movies, scenes) for behavior session.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the stimulus images presented during the
            session. Keys are data set names, and values are 3D numpy arrays.
        """
        return self.api.get_stimulus_templates()

    @property
    def stimulus_timestamps(self) -> np.ndarray:
        """Get stimulus timestamps from pkl file.

        NOTE: Located with behavior_session_id

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
            that do no account for monitor delay.
        """
        return self.api.get_stimulus_timestamps()

    @property
    def task_parameters(self) -> dict:
        """Get task parameters from pkl file.

        Returns
        -------
        dict
            A dictionary containing parameters used to define the task runtime
            behavior.
        """
        return self.api.get_task_parameters()

    @property
    def trials(self) -> pd.DataFrame:
        """Get trials from pkl file

        Returns
        -------
        pd.DataFrame
            A dataframe containing behavioral trial start/stop times,
            and trial data
        """
        return self.api.get_trials()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the session.
        :rtype: dict
        """
        return self.api.get_metadata()

    def cache_clear(self) -> None:
        """Convenience method to clear the api cache, if applicable."""
        try:
            self.api.cache_clear()
        except AttributeError:
            logging.getLogger("BehaviorOphysSession").warning(
                "Attempted to clear API cache, but method `cache_clear`"
                f" does not exist on {self.api.__class__.__name__}")

    def list_api_methods(self) -> List[Tuple[str, str]]:
        """Convenience method to expose list of API `get` methods. These methods
        can be accessed by referencing the API used to initialize this
        BehaviorDataSession via its `api` instance attribute.
        :rtype: list of tuples, where the first value in the tuple is the
        method name, and the second value is the method docstring.
        """
        methods = [m for m in inspect.getmembers(self.api, inspect.ismethod)
                   if m[0].startswith("get_")]
        docs = [inspect.getdoc(m[1]) or "" for m in methods]
        method_names = [m[0] for m in methods]
        return list(zip(method_names, docs))
