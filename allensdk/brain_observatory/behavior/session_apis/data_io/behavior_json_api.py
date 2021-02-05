import logging
from datetime import datetime
import pytz

from allensdk.brain_observatory.behavior.session_apis.data_transforms import (
    BehaviorDataXforms)


class BehaviorJsonApi(BehaviorDataXforms):
    """A data fetching class that serves as an API for fetching 'raw'
    data from a json file necessary (but not sufficient) for filling
    a 'BehaviorSession'.

    Most 'raw' data provided by this API needs to be processed by
    BehaviorDataXforms methods in order to usable by 'BehaviorSession's.

    This class is used by the write_nwb module for behavior sessions.
    """

    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_behavior_session_id(self) -> int:
        return self.data['behavior_session_id']

    def get_foraging_id(self) -> int:
        return self.data['foraging_id']

    def get_rig_name(self) -> str:
        """Get the name of the experiment rig (ex: CAM2P.3)"""
        return self.data['rig_name']

    def get_sex(self) -> str:
        """Get the sex of the subject (ex: 'M', 'F', or 'unknown')"""
        return self.data['sex']

    def get_age(self) -> str:
        """Get the age of the subject (ex: 'P15', 'Adult', etc...)"""
        return self.data['age']

    def get_stimulus_name(self) -> str:
        """Get the name of the stimulus presented for a behavior or
        behavior + ophys experiment"""
        return self.data['stimulus_name']

    def get_experiment_date(self) -> datetime:
        """Get the acquisition date of an ophys experiment"""
        return pytz.utc.localize(
            datetime.strptime(self.data['date_of_acquisition'],
                              "%Y-%m-%d %H:%M:%S"))

    def get_reporter_line(self) -> str:
        """Get the (gene) reporter line for the subject associated with a
        behavior or behavior + ophys experiment"""
        return self.data['reporter_line']

    def get_driver_line(self) -> str:
        """Get the (gene) driver line for the subject associated with a
        behavior or behavior + ophys experiment"""
        return self.data['driver_line']

    def get_full_genotype(self) -> str:
        """Get the full genotype of the subject associated with a
        behavior or behavior + ophys experiment"""
        return self.data['full_genotype']

    def get_behavior_stimulus_file(self) -> str:
        """Get the filepath to the StimulusPickle file for the session"""
        return self.data['behavior_stimulus_file']

    def get_external_specimen_name(self) -> int:
        """Get the external specimen id (LabTracks ID) for the subject
        associated with a behavior experiment"""
        return int(self.data['external_specimen_name'])
