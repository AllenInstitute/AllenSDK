from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces import \
    NwbReadableInterface


class EmissionLambda(DataObject, NwbReadableInterface):
    def __init__(self, emission_lambda=520.0):
        super().__init__(name='emission_lambda', value=emission_lambda)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "EmissionLambda":
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        imaging_plane = image_seg.plane_segmentations[
            'cell_specimen_table'].imaging_plane
        optical_channel = imaging_plane.optical_channel[0]
        emission_lambda = optical_channel.emission_lambda
        return cls(emission_lambda=emission_lambda)
