from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects._base.readable_mixins\
    .stimulus_file_readable_mixin \
    import \
    StimulusFileReadableMixin


class StimulusFrameRate(DataObject, StimulusFileReadableMixin):
    """Stimulus frame rate"""
    def __init__(self, stimulus_frame_rate: float):
        super().__init__(name="stimulus_frame_rate", value=stimulus_frame_rate)

    def to_json(self) -> dict:
        return {"stimulus_frame_rate": self.value}

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_timestamps: StimulusTimestamps) -> "StimulusFrameRate":
        frame_rate = stimulus_timestamps.calc_frame_rate()
        return cls(stimulus_frame_rate=frame_rate)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "StimulusFrameRate":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(stimulus_frame_rate=metadata.stimulus_frame_rate)
