import abc
from datetime import datetime
from typing import List


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
    def get_rig_name(self) -> str:
        """Get the name of the experiment rig (ex: CAM2P.3)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_sex(self) -> str:
        """Get the sex of the subject (ex: 'M', 'F', or 'unknown')"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_age(self) -> str:
        """Get the age of the subject (ex: 'P15', 'Adult', etc...)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stimulus_name(self) -> str:
        """Get the name of the stimulus presented for a behavior or
        behavior + ophys experiment"""
        raise NotImplementedError()

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
    def get_external_specimen_name(self) -> int:
        """Get the external specimen id (LabTracks ID) for the subject
        associated with a behavior experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_experiment_date(self) -> datetime:
        """Get the acquisition date of an experiment in UTC"""
        raise NotImplementedError()
