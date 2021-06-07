
import json
from typing import Optional

from cachetools.keys import hashkey

import numpy as np
from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries

from allensdk.brain_observatory.behavior.data_objects._base.readable_interfaces\
    .stimulus_file_readable_interface \
    import \
    StimulusFileReadableInterface
from allensdk.brain_observatory.behavior.data_objects._base.readable_interfaces\
    .sync_file_readable_interface import \
    SyncFileReadableInterface
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_files import (
    StimulusFile, SyncFile
)
from allensdk.brain_observatory.behavior.data_objects._base\
    .writable_interfaces.json_writable_interface import \
    JsonWritableInterface
from allensdk.brain_observatory.behavior.data_objects._base\
    .writable_interfaces.nwb_writable_interface import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.stimulus_timestamps.timestamps_processing import (  # noqa: E501
    get_behavior_stimulus_timestamps, get_ophys_stimulus_timestamps
)


def from_json_cache_key(cls, dict_repr: dict):
    return hashkey(json.dumps(dict_repr))


def from_lims_cache_key(
    cls, db, behavior_session_id: int,
    ophys_experiment_id: Optional[int] = None
):
    return hashkey(behavior_session_id, ophys_experiment_id)


class StimulusTimestamps(DataObject, StimulusFileReadableInterface,
                         SyncFileReadableInterface, NwbWritableInterface,
                         JsonWritableInterface):
    """A DataObject which contains properties and methods to load, process,
    and represent visual behavior stimulus timestamp data.

    Stimulus timestamp data is represented as:

    Numpy array whose length is equal to the number of timestamps collected
    and whose values are timestamps (in seconds)
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        stimulus_file: Optional[StimulusFile] = None,
        sync_file: Optional[SyncFile] = None
    ):
        super().__init__(name="stimulus_timestamps", value=timestamps)
        self._stimulus_file = stimulus_file
        self._sync_file = sync_file

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_file: StimulusFile) -> "StimulusTimestamps":
        stimulus_timestamps = get_behavior_stimulus_timestamps(
            stimulus_pkl=stimulus_file.data
        )

        return cls(
            timestamps=stimulus_timestamps,
            stimulus_file=stimulus_file
        )

    @classmethod
    def from_sync_file(cls, sync_file: SyncFile) -> "StimulusTimestamps":
        stimulus_timestamps = get_ophys_stimulus_timestamps(
            sync_path=sync_file.filepath
        )
        return cls(
            timestamps=stimulus_timestamps,
            sync_file=sync_file
        )

    def to_json(self) -> dict:
        if self._stimulus_file is None:
            raise RuntimeError(
                "StimulusTimestamps DataObject lacks information about the "
                "StimulusFile. This is likely due to instantiating from NWB "
                "which prevents to_json() functionality"
            )

        output_dict = dict()
        output_dict.update(self._stimulus_file.to_json())
        if self._sync_file is not None:
            output_dict.update(self._sync_file.to_json())
        return output_dict

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "StimulusTimestamps":
        stim_module = nwbfile.processing["stimulus"]
        stim_ts_interface = stim_module.get_data_interface("timestamps")
        stim_timestamps = stim_ts_interface.timestamps[:]
        return cls(timestamps=stim_timestamps)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        stimulus_ts = TimeSeries(
            data=self._value,
            name="timestamps",
            timestamps=self._value,
            unit="s"
        )

        stim_mod = ProcessingModule("stimulus", "Stimulus Times processing")
        stim_mod.add_data_interface(stimulus_ts)
        nwbfile.add_processing_module(stim_mod)

        return nwbfile

    def calc_frame_rate(self):
        return np.round(1 / np.mean(np.diff(self.value)), 0)
