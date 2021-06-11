from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces.nwb_readable_interface import \
    NwbReadableInterface


class ExcitationLambda(DataObject, NwbReadableInterface):
    def __init__(self, excitation_lambda=910.0):
        super().__init__(name='excitation_lambda', value=excitation_lambda)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "ExcitationLambda":
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        imaging_plane = image_seg.plane_segmentations[
            'cell_specimen_table'].imaging_plane
        excitation_lambda = imaging_plane.excitation_lambda
        return cls(excitation_lambda=excitation_lambda)
