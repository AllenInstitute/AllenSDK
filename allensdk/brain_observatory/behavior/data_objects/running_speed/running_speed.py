from typing import Optional

import numpy as np
import pandas as pd
from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries

from allensdk.brain_observatory.behavior.data_files import (
    BehaviorStimulusFile, MappingStimulusFile, ReplayStimulusFile, SyncFile)
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.running_speed.multi_stim_running_processing import \
    _get_multi_stim_running_df
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import \
    get_running_df  # noqa: E501
from allensdk.core import (DataObject, NwbReadableInterface,
                           NwbWritableInterface)
from allensdk.core.exceptions import DataFrameIndexError


class RunningSpeed(DataObject,
                   NwbReadableInterface,
                   NwbWritableInterface):
    """A DataObject which contains properties and methods to load, process,
    and represent running speed data.

    Running speed data is represented as:

    Pandas Dataframe with the following columns:
        "timestamps": Timestamps (in s) for calculated speed values
        "speed": Computed running speed in cm/s
    """

    def __init__(
        self,
        running_speed: pd.DataFrame,
        stimulus_file: Optional[BehaviorStimulusFile] = None,
        sync_file: Optional[SyncFile] = None,
        stimulus_timestamps: Optional[StimulusTimestamps] = None,
        filtered: bool = True
    ):
        super().__init__(name='running_speed', value=running_speed)

        if stimulus_timestamps is not None:
            if not np.isclose(stimulus_timestamps.monitor_delay, 0.0):
                raise RuntimeError(
                    "Running speed timestamps have monitor delay "
                    f"{stimulus_timestamps.monitor_delay}; there "
                    "should be no monitor delay applied to the timestamps "
                    "associated with running speed")

        self._stimulus_file = stimulus_file
        self._sync_file = sync_file
        self._stimulus_timestamps = stimulus_timestamps
        self._filtered = filtered

    @staticmethod
    def _get_running_speed_df(
        stimulus_file: BehaviorStimulusFile,
        stimulus_timestamps: StimulusTimestamps,
        filtered: bool = True,
        zscore_threshold: float = 1.0
    ) -> pd.DataFrame:
        running_data_df = get_running_df(
            data=stimulus_file.data, time=stimulus_timestamps.value,
            lowpass=filtered, zscore_threshold=zscore_threshold
        )
        if running_data_df.index.name != "timestamps":
            raise DataFrameIndexError(
                f"Expected running_data_df index to be named 'timestamps' "
                f"But instead got: '{running_data_df.index.name}'"
            )
        running_speed = pd.DataFrame({
            "timestamps": running_data_df.index.values,
            "speed": running_data_df.speed.values
        })
        return running_speed

    @classmethod
    def from_stimulus_file(
            cls,
            behavior_stimulus_file: BehaviorStimulusFile,
            sync_file: Optional[SyncFile] = None,
            filtered: bool = True,
            zscore_threshold: float = 10.0) -> "RunningSpeed":
        """
        sync_file is used for generating timestamps. If None, timestamps
        will be generated from the stimulus file.
        """

        if sync_file is not None:
            stimulus_timestamps = StimulusTimestamps.from_sync_file(
                                        sync_file=sync_file,
                                        monitor_delay=0.0)

        else:
            stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
                                        stimulus_file=behavior_stimulus_file,
                                        monitor_delay=0.0)

        running_speed = cls._get_running_speed_df(
            behavior_stimulus_file,
            stimulus_timestamps,
            filtered,
            zscore_threshold
        )
        return cls(
            running_speed=running_speed,
            stimulus_file=behavior_stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            filtered=filtered)

    @classmethod
    def from_multiple_stimulus_files(
            cls,
            behavior_stimulus_file: BehaviorStimulusFile,
            mapping_stimulus_file: MappingStimulusFile,
            replay_stimulus_file: ReplayStimulusFile,
            sync_file: SyncFile,
            filtered: bool = True,
            zscore_threshold: float = 10.0) -> "RunningSpeed":
        """
        sync_file is used for generating timestamps.

        Stimulus blocks are assumed to be presented in the order
        behavior_stimulus_file
        mapping_stimulus_file
        replay_stimulus_file
        """

        df = _get_multi_stim_running_df(
                sync_path=sync_file.filepath,
                behavior_stimulus_file=behavior_stimulus_file,
                mapping_stimulus_file=mapping_stimulus_file,
                replay_stimulus_file=replay_stimulus_file,
                use_lowpass_filter=filtered,
                zscore_threshold=zscore_threshold)['running_speed']

        return cls(
               running_speed=df,
               filtered=filtered,
               sync_file=None,
               stimulus_file=None,
               stimulus_timestamps=None)

    @classmethod
    def from_nwb(
        cls,
        nwbfile: NWBFile,
        filtered=True
    ) -> "RunningSpeed":
        running_module = nwbfile.modules['running']
        interface_name = 'speed' if filtered else 'speed_unfiltered'
        running_interface = running_module.get_data_interface(interface_name)

        timestamps = running_interface.timestamps[:]
        values = running_interface.data[:]

        running_speed = pd.DataFrame(
            {
                "timestamps": timestamps,
                "speed": values
            }
        )
        return cls(running_speed=running_speed, filtered=filtered)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        running_speed: pd.DataFrame = self.value
        data = running_speed['speed'].values
        timestamps = running_speed['timestamps'].values

        if self._filtered:
            data_interface_name = "speed"
        else:
            data_interface_name = "speed_unfiltered"

        running_speed_series = TimeSeries(
            name=data_interface_name,
            data=data,
            timestamps=timestamps,
            unit='cm/s')

        if 'running' in nwbfile.processing:
            running_mod = nwbfile.processing['running']
        else:
            running_mod = ProcessingModule('running',
                                           'Running speed processing module')
            nwbfile.add_processing_module(running_mod)

        running_mod.add_data_interface(running_speed_series)

        return nwbfile
