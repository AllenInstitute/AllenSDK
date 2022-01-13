from typing import Optional

import numpy as np
from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries

from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface, \
    StimulusFileReadableInterface, SyncFileReadableInterface
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_files import (
    StimulusFile, SyncFile
)
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    JsonWritableInterface, NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects.timestamps\
    .stimulus_timestamps.timestamps_processing import (
        get_behavior_stimulus_timestamps, get_ophys_stimulus_timestamps)
from allensdk.internal.api import PostgresQueryMixin


class StimulusTimestamps(DataObject, StimulusFileReadableInterface,
                         SyncFileReadableInterface, JsonReadableInterface,
                         NwbReadableInterface, LimsReadableInterface,
                         NwbWritableInterface, JsonWritableInterface):
    """A DataObject which contains properties and methods to load, process,
    and represent visual behavior stimulus timestamp data.

    Stimulus timestamp data is represented as:

    Numpy array whose length is equal to the number of timestamps collected
    and whose values are timestamps (in seconds)
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        monitor_delay: float,
        stimulus_file: Optional[StimulusFile] = None,
        sync_file: Optional[SyncFile] = None
    ):
        super().__init__(name="stimulus_timestamps",
                         value=timestamps+monitor_delay)
        self._stimulus_file = stimulus_file
        self._sync_file = sync_file
        self._monitor_delay = monitor_delay

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_file: StimulusFile,
            monitor_delay: float) -> "StimulusTimestamps":
        stimulus_timestamps = get_behavior_stimulus_timestamps(
            stimulus_pkl=stimulus_file.data
        )

        return cls(
            timestamps=stimulus_timestamps,
            monitor_delay=monitor_delay,
            stimulus_file=stimulus_file
        )

    @classmethod
    def from_sync_file(
            cls,
            sync_file: SyncFile,
            monitor_delay: float) -> "StimulusTimestamps":
        stimulus_timestamps = get_ophys_stimulus_timestamps(
            sync_path=sync_file.filepath
        )
        return cls(
            timestamps=stimulus_timestamps,
            monitor_delay=monitor_delay,
            sync_file=sync_file
        )

    @classmethod
    def from_json(
            cls,
            dict_repr: dict) -> "StimulusTimestamps":
        if 'sync_file' in dict_repr:
            sync_file = SyncFile.from_json(dict_repr=dict_repr)
            return cls.from_sync_file(
                        sync_file=sync_file,
                        monitor_delay=dict_repr['monitor_delay'])
        else:
            stim_file = StimulusFile.from_json(dict_repr=dict_repr)
            return cls.from_stimulus_file(
                        stimulus_file=stim_file,
                        monitor_delay=dict_repr['monitor_delay'])

    def from_lims(
        cls,
        db: PostgresQueryMixin,
        monitor_delay: float,
        behavior_session_id: int,
        ophys_experiment_id: Optional[int] = None
    ) -> "StimulusTimestamps":
        stimulus_file = StimulusFile.from_lims(db, behavior_session_id)

        if ophys_experiment_id:
            sync_file = SyncFile.from_lims(
                db=db, ophys_experiment_id=ophys_experiment_id)
            return cls.from_sync_file(sync_file=sync_file,
                                      monitor_delay=monitor_delay)
        else:
            return cls.from_stimulus_file(stimulus_file=stimulus_file,
                                          monitor_delay=monitor_delay)

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
        output_dict['monitor_delay'] = self._monitor_delay
        return output_dict

    @classmethod
    def from_nwb(cls,
                 nwbfile: NWBFile) -> "StimulusTimestamps":
        stim_module = nwbfile.processing["stimulus"]
        stim_ts_interface = stim_module.get_data_interface("timestamps")
        stim_timestamps = stim_ts_interface.timestamps[:]

        # Because the monitor delay was already applied when
        # saving the stimulus timestamps to the NWB file,
        # we set it to zero here.
        return cls(timestamps=stim_timestamps,
                   monitor_delay=0.0)

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
