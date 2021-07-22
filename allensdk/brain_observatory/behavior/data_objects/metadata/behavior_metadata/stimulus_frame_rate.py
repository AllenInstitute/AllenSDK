from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    NwbReadableInterface, StimulusFileReadableInterface


class StimulusFrameRate(DataObject, StimulusFileReadableInterface,
                        NwbReadableInterface):
    """Stimulus frame rate"""
    def __init__(self, stimulus_frame_rate: float):
        super().__init__(name="stimulus_frame_rate", value=stimulus_frame_rate)

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_file: StimulusFile) -> "StimulusFrameRate":
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file)
        frame_rate = stimulus_timestamps.calc_frame_rate()
        return cls(stimulus_frame_rate=frame_rate)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "StimulusFrameRate":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(stimulus_frame_rate=metadata.stimulus_frame_rate)
