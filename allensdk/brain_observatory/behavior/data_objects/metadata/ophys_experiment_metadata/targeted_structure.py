from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.json_readable_interface import \
    JsonReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.lims_readable_interface import \
    LimsReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces.nwb_readable_interface import \
    NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class TargetedStructure(DataObject, LimsReadableInterface,
                        JsonReadableInterface, NwbReadableInterface):
    """targeted structure (acronym) for an ophys experiment
        (ex: "Visp")"""
    def __init__(self, targeted_structure: str):
        super().__init__(name='targeted_structure', value=targeted_structure)

    @classmethod
    def from_lims(cls, ophys_experiment_id,
                  lims_db: PostgresQueryMixin) -> "TargetedStructure":
        query = """
                SELECT st.acronym
                FROM ophys_experiments oe
                LEFT JOIN structures st ON st.id = oe.targeted_structure_id
                WHERE oe.id = {};
                """.format(ophys_experiment_id)
        targeted_structure = lims_db.fetchone(query, strict=True)
        return cls(targeted_structure=targeted_structure)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "TargetedStructure":
        return cls(targeted_structure=dict_repr['targeted_structure'])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "TargetedStructure":
        ophys_module = nwbfile.processing['ophys']
        image_seg = ophys_module.data_interfaces['image_segmentation']
        imaging_plane = image_seg.plane_segmentations[
            'cell_specimen_table'].imaging_plane
        targeted_structure = imaging_plane.location
        return cls(targeted_structure=targeted_structure)
