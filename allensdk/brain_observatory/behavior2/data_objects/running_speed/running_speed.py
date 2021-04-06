import json

import pandas as pd
import pynwb
from pynwb import NWBFile, ProcessingModule

from allensdk.brain_observatory.behavior2.data_object import \
    DataObject
from allensdk.brain_observatory.behavior2.data_objects.running_speed\
    .running_processing import \
    get_running_df
from allensdk.brain_observatory.behavior2.stimulus_file import StimulusFile


class RunningSpeed(DataObject):
    def __init__(self, running_speed: pd.DataFrame):
        super().__init__(name='running_speed', value=running_speed)

    @staticmethod
    def from_stimulus_file(stimulus_file: StimulusFile,
                           lowpass=True) -> "RunningSpeed":
        running_data_df = RunningSpeed._get_running_data_df(
            lowpass=lowpass, stimulus_file=stimulus_file)
        running_speed = pd.DataFrame({
            "timestamps": running_data_df.index.values,
            "values": running_data_df.speed.values})
        return RunningSpeed(running_speed=running_speed)

    def from_lims(self):
        raise NotImplementedError()

    @staticmethod
    def from_json(dict_repr: dict) -> "RunningSpeed":
        running_speed = json.loads(dict_repr['running_speed'])
        running_speed = pd.DataFrame(running_speed)
        return RunningSpeed(running_speed=running_speed)

    def from_nwb(self):
        pass

    def to_json(self):
        value: pd.DataFrame = self._value

        # shorten for example
        value = value.iloc[:3]

        return {self._name: value.to_json(orient='records')}

    def to_nwb(self, nwbfile: NWBFile):
        running_speed: pd.DataFrame = self._value
        data = running_speed['values'].values
        timestamps = running_speed['timestamps'].values

        running_speed_series = pynwb.base.TimeSeries(
            name=self._name,
            data=data,
            timestamps=timestamps,
            unit='m/s')

        if 'running' in nwbfile.processing:
            running_mod = nwbfile.processing['running']
        else:
            running_mod = ProcessingModule('running',
                                           'Running speed processing module')
            nwbfile.add_processing_module(running_mod)

        running_mod.add_data_interface(running_speed_series)

    @staticmethod
    def _get_running_data_df(stimulus_file: StimulusFile,
                             lowpass=True,
                             zscore_threshold=10.0) -> pd.DataFrame:
        """Get running speed data.

        :returns: pd.DataFrame -- dataframe containing various signals used
            to compute running speed.
        """
        stimulus_timestamps = stimulus_file.get_stimulus_timestamps()
        data = stimulus_file.data
        return get_running_df(data=data, time=stimulus_timestamps,
                              lowpass=lowpass,
                              zscore_threshold=zscore_threshold)
