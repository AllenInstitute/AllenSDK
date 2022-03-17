from typing import Optional

import pandas as pd

from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries

from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    JsonWritableInterface, NwbWritableInterface
from allensdk.core.exceptions import DataFrameIndexError
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_objects import (
    DataObject, StimulusTimestamps
)
from allensdk.brain_observatory.behavior.data_files import (
    StimulusFile
)
# from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
#     get_running_df
# )

from allensdk.brain_observatory.ecephys.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)

from allensdk.brain_observatory.ecephys.data_files.ecephys_running_speed_file import EcephysRunningSpeedFile

class RunningSpeed(DataObject, LimsReadableInterface, NwbReadableInterface,
                   NwbWritableInterface, JsonReadableInterface,
                   JsonWritableInterface):
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
        stimulus_file: Optional[StimulusFile] = None,
        stimulus_timestamps: Optional[StimulusTimestamps] = None,
        filtered: bool = True
    ):
        super().__init__(name='running_speed', value=running_speed)
        self._stimulus_file = stimulus_file
        self._stimulus_timestamps = stimulus_timestamps
        self._filtered = filtered

    @staticmethod
    def _get_running_speed_df(
        ecephys_running_speed_file: EcephysRunningSpeedFile,
        stimulus_timestamps: StimulusTimestamps,
        filtered: bool = True,
        zscore_threshold: float = 1.0
    ) -> pd.DataFrame:

        # print('stimulus_timestamps.value', str(len(stimulus_timestamps.value)))
        # print('behavior_stimulus_file.data')
        

        running_data_df = get_running_df(
            data=ecephys_running_speed_file.data, 
            time=stimulus_timestamps.value,
            lowpass=filtered, zscore_threshold=zscore_threshold
        )


        if running_data_df.index.name != "timestamps":
            raise DataFrameIndexError(
                f"Expected running_data_df index to be named 'timestamps' "
                f"But instead got: '{running_data_df.index.name}'"
            )
        running_speed = pd.DataFrame({
            "timestamps": running_data_df.index.values,
            "speed": running_data_df.speed.values,
            "dx": running_data_df.dx.values
        })
        return running_speed

    @classmethod
    def from_json(
        cls,
        dict_repr: dict,
        filtered: bool = True,
        zscore_threshold: float = 10.0
    ) -> "RunningSpeed":
        stimulus_file = StimulusFile.from_json(dict_repr)
        stimulus_timestamps = StimulusTimestamps.from_json(dict_repr)


        running_speed_path = dict_repr['running_speed_path']

        if not filtered:
            #TODO uncomment this
            # running_speed_path = dict_repr['raw_running_speed_path']
            running_speed_path = dict_repr['running_speed_path']
        

        ecephys_running_speed_file = EcephysRunningSpeedFile.from_json(dict_repr=dict_repr)


        running_speed = cls._get_running_speed_df(
            ecephys_running_speed_file, stimulus_timestamps, filtered, zscore_threshold
        )
        return cls(
            running_speed=running_speed,
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            filtered=filtered)

    def to_json(self) -> dict:
        if self._stimulus_file is None or self._stimulus_timestamps is None:
            raise RuntimeError(
                "RunningSpeed DataObject lacks information about the "
                "StimulusFile or StimulusTimestamps. This is likely due to "
                "instantiating from NWB which prevents to_json() functionality"
            )
        output_dict = dict()
        output_dict.update(self._stimulus_file.to_json())
        output_dict.update(self._stimulus_timestamps.to_json())
        return output_dict

    @classmethod
    def from_lims(
        cls,
        db: PostgresQueryMixin,
        behavior_session_id: int,
        filtered: bool = True,
        zscore_threshold: float = 10.0,
        stimulus_timestamps: Optional[StimulusTimestamps] = None,
        monitor_delay: Optional[float] = None,
    ) -> "RunningSpeed":
        stimulus_file = StimulusFile.from_lims(db, behavior_session_id)
        if stimulus_timestamps is None:
            if monitor_delay is None:
                raise RuntimeError("Must specify monitor delay "
                                   "if you are not specifying "
                                   "stimulus_timestamps")
            stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
                stimulus_file=stimulus_file,
                monitor_delay=monitor_delay
            )

        running_speed = cls._get_running_speed_df(
            stimulus_file, stimulus_timestamps, filtered, zscore_threshold
        )
        return cls(
            running_speed=running_speed,
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            filtered=filtered
        )

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

        #only need to add this once
        if self._filtered:
            dx_running_speed_series = TimeSeries(
            name='dx',
            data=running_speed['dx'].values,
            timestamps=timestamps,
            unit='cm/s')
            running_mod.add_data_interface(dx_running_speed_series)

        return nwbfile
