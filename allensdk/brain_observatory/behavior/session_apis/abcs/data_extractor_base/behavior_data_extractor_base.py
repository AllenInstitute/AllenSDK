import abc
import pandas as pd
from datetime import datetime
from typing import List

from allensdk.api.warehouse_cache.cache import memoize


class BehaviorDataExtractorBase(abc.ABC):
    """Abstract base class implementing required methods for extracting
    data (from LIMS or from JSON) that will be transformed or passed on to
    fill behavior session data.
    """

    @abc.abstractmethod
    def get_behavior_session_id(self) -> int:
        """Get the ID of the behavior session"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_foraging_id(self) -> int:
        """Get the foraging ID for the behavior session"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_equipment_name(self) -> str:
        """Get the name of the experiment rig (ex: CAM2P.3)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_sex(self) -> str:
        """Get the sex of the subject (ex: 'M', 'F', or 'unknown')"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_age(self) -> str:
        """Get the age code of the subject (ie P123)"""
        raise NotImplementedError()

    @memoize
    def get_stimulus_name(self) -> str:
        """Get the stimulus set used from the behavior session pkl file
        :rtype: str
        """
        behavior_stimulus_path = self.get_behavior_stimulus_file()
        pkl = pd.read_pickle(behavior_stimulus_path)

        try:
            stimulus_name = pkl["items"]["behavior"]["cl_params"]["stage"]
        except KeyError:
            raise RuntimeError(
                f"Could not obtain stimulus_name/stage information from "
                f"the *.pkl file ({behavior_stimulus_path}) "
                f"for the behavior session to save as NWB! The "
                f"following series of nested keys did not work: "
                f"['items']['behavior']['cl_params']['stage']"
            )
        return stimulus_name

    @abc.abstractmethod
    def get_reporter_line(self) -> List[str]:
        """Get the (gene) reporter line(s) for the subject associated with a
        behavior or behavior + ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_driver_line(self) -> List[str]:
        """Get the (gene) driver line(s) for the subject associated with a
        behavior or behavior + ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_full_genotype(self) -> str:
        """Get the full genotype of the subject associated with a
        behavior or behavior + ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_behavior_stimulus_file(self) -> str:
        """Get the filepath to the StimulusPickle file for the session"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_mouse_id(self) -> int:
        """Get the mouse id (LabTracks ID) for the subject
        associated with a behavior experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_date_of_acquisition(self) -> datetime:
        """Get the acquisition date of an experiment in UTC"""
        raise NotImplementedError()
