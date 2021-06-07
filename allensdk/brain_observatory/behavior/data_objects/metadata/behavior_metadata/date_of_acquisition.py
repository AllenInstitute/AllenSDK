import warnings
from datetime import datetime

import pytz
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects._base\
    .readable_interfaces.json_readable_interface import \
    JsonReadableInterface
from allensdk.brain_observatory.behavior.data_objects._base.readable_interfaces\
    .lims_readable_interface import \
    LimsReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class DateOfAcquisition(DataObject, LimsReadableInterface,
                        JsonReadableInterface):
    """timestamp for when experiment was started in UTC"""
    def __init__(self, date_of_acquisition: float):
        super().__init__(name="date_of_acquisition", value=date_of_acquisition)

    def from_json(cls, dict_repr: dict) -> "DateOfAcquisition":
        pass

    def to_json(self) -> dict:
        return {"stimulus_frame_rate": self.value}

    @classmethod
    def from_lims(
            cls, behavior_session_id: int,
            lims_db: PostgresQueryMixin) -> "DateOfAcquisition":
        query = """
                SELECT bs.date_of_acquisition
                FROM behavior_sessions bs
                WHERE bs.id = {};
                """.format(behavior_session_id)

        experiment_date = lims_db.fetchone(query, strict=True)
        experiment_date = pytz.utc.localize(experiment_date)
        return cls(date_of_acquisition=experiment_date)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "DateOfAcquisition":
        return cls(date_of_acquisition=nwbfile.session_start_time)

    def validate(self, stimulus_file: StimulusFile,
                 behavior_session_id: int) -> "DateOfAcquisition":
        """raise a warning if the date differs too much from the
        datetime obtained from the behavior stimulus (*.pkl) file."""
        pkl_data = stimulus_file.data
        pkl_raw_acq_date = pkl_data["start_time"]
        if isinstance(pkl_raw_acq_date, datetime):
            pkl_acq_date = pytz.utc.localize(pkl_raw_acq_date)

        elif isinstance(pkl_raw_acq_date, (int, float)):
            # We are dealing with an older pkl file where the acq time is
            # stored as a Unix style timestamp string
            parsed_pkl_acq_date = datetime.fromtimestamp(pkl_raw_acq_date)
            pkl_acq_date = pytz.utc.localize(parsed_pkl_acq_date)
        else:
            pkl_acq_date = None
            warnings.warn(
                "Could not parse the acquisition datetime "
                f"({pkl_raw_acq_date}) found in the following stimulus *.pkl: "
                f"{stimulus_file.filepath}"
            )

        if pkl_acq_date:
            acq_start_diff = (
                    self.value - pkl_acq_date).total_seconds()
            # If acquisition dates differ by more than an hour
            if abs(acq_start_diff) > 3600:
                session_id = behavior_session_id
                warnings.warn(
                    "The `date_of_acquisition` field in LIMS "
                    f"({self.value}) for behavior session "
                    f"({session_id}) deviates by more "
                    f"than an hour from the `start_time` ({pkl_acq_date}) "
                    "specified in the associated stimulus *.pkl file: "
                    f"{stimulus_file.filepath}"
                )
        return self
