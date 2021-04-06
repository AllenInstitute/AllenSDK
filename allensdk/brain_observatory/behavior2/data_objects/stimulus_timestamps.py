import numpy as np
from pynwb import TimeSeries, ProcessingModule, NWBFile

from allensdk.brain_observatory.behavior2.data_object import \
    DataObject
from allensdk.brain_observatory.behavior2.stimulus_file import StimulusFile


class StimulusTimestamps(DataObject):
    def __init__(self, timestamps: np.array):
        super().__init__(name='stimulus_timestamps', value=timestamps)

    @staticmethod
    def from_stimulus_file(
            stimulus_file: StimulusFile) -> "StimulusTimestamps":
        timestamps = stimulus_file.get_stimulus_timestamps()
        return StimulusTimestamps(timestamps=timestamps)

    def from_lims(self):
        raise NotImplementedError()

    @staticmethod
    def from_json(dict_repr: dict) -> "StimulusTimestamps":
        stimulus_timestamps = np.array(dict_repr['stimulus_timestamps'])
        return StimulusTimestamps(timestamps=stimulus_timestamps)

    def from_nwb(self):
        pass

    def to_json(self):
        return {self._name: self._value.tolist()[:5]}

    def to_nwb(self, nwbfile: NWBFile):
        stimulus_ts = TimeSeries(
            data=self._value,
            name='timestamps',
            timestamps=self._value,
            unit='s'
        )

        stim_mod = ProcessingModule('stimulus', 'Stimulus Times processing')

        nwbfile.add_processing_module(stim_mod)
        stim_mod.add_data_interface(stimulus_ts)
