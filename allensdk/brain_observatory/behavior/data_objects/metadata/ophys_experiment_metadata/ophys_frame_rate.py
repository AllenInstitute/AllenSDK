from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.nwb_readable_interface import \
    NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base.readable_interfaces\
    .stimulus_file_readable_interface \
    import \
    StimulusFileReadableInterface


class OphysFrameRate(DataObject, StimulusFileReadableInterface,
                        NwbReadableInterface):
    """Ophys frame rate"""
    def __init__(self, ophys_frame_rate: float):
        super().__init__(name="ophys_frame_rate", value=ophys_frame_rate)

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_timestamps: StimulusTimestamps) -> "OphysFrameRate":
        frame_rate = stimulus_timestamps.calc_frame_rate()
        return cls(ophys_frame_rate=frame_rate)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "OphysFrameRate":
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        imaging_plane = image_seg.plane_segmentations[
            'cell_specimen_table'].imaging_plane
        ophys_frame_rate = imaging_plane.imaging_rate
        return cls(ophys_frame_rate=ophys_frame_rate)