import logging
from datetime import datetime

import pytz
from allensdk.brain_observatory.behavior.session_apis.abcs. \
    data_extractor_base.behavior_data_extractor_base import \
    BehaviorDataExtractorBase
from allensdk.brain_observatory.behavior.session_apis.data_transforms import \
    BehaviorDataTransforms


class BehaviorJsonApi(BehaviorDataTransforms):
    """A data fetching and processing class that serves processed data from
    a specified raw data source (extractor). Contains all methods
    needed to fill a BehaviorSession."""

    def __init__(self, data):
        extractor = BehaviorJsonExtractor(data=data)
        super().__init__(extractor=extractor)


class BehaviorJsonExtractor(BehaviorDataExtractorBase):
    """A class which 'extracts' data from a json file. The extracted data
    is necessary (but not sufficient) for populating a 'BehaviorSession'.

    Most data provided by this extractor needs to be processed by
    BehaviorDataTransforms methods in order to usable by 'BehaviorSession's.

    This class is used by the write_nwb module for behavior sessions.
    """

    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_behavior_session_id(self) -> int:
        return self.data['behavior_session_id']

    def get_foraging_id(self) -> int:
        return self.data['foraging_id']

    def get_equipment_name(self) -> str:
        """Get the name of the experiment rig (ex: CAM2P.3)"""
        return self.data['rig_name']

    def get_sex(self) -> str:
        """Get the sex of the subject (ex: 'M', 'F', or 'unknown')"""
        return self.data['sex']

    def get_age(self) -> str:
        """Get the age code of the subject (ie P123)"""
        return self.data['age']

    def get_reporter_line(self) -> str:
        """Get the (gene) reporter line for the subject associated with an
        experiment"""
        return self.data['reporter_line']

    def get_driver_line(self) -> str:
        """Get the (gene) driver line for the subject associated with an
        experiment"""
        return self.data['driver_line']

    def get_full_genotype(self) -> str:
        """Get the full genotype of the subject associated with an
        experiment"""
        return self.data['full_genotype']

    def get_behavior_stimulus_file(self) -> str:
        """Get the filepath to the StimulusPickle file for the session"""
        return self.data['behavior_stimulus_file']

    def get_mouse_id(self) -> int:
        """Get the external specimen id (LabTracks ID) for the subject
        associated with a behavior experiment"""
        return int(self.data['external_specimen_name'])

    def get_date_of_acquisition(self) -> datetime:
        """Get the acquisition date of an experiment (in UTC)

        NOTE: LIMS writes to JSON in local time. Needs to be converted to UTC
        """
        tz = pytz.timezone("America/Los_Angeles")
        return tz.localize(datetime.strptime(self.data['date_of_acquisition'],
                                             "%Y-%m-%d %H:%M:%S")).astimezone(
            pytz.utc)
