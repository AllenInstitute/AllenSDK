from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class ImagingDepth(DataObject, LimsReadableInterface, NwbReadableInterface,
                   JsonReadableInterface):
    def __init__(self, imaging_depth: int):
        super().__init__(name='imaging_depth', value=imaging_depth)

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin) -> "ImagingDepth":
        query = """
                SELECT id.depth
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN imaging_depths id ON id.id = oe.imaging_depth_id
                WHERE oe.id = {};
                """.format(ophys_experiment_id)
        imaging_depth = lims_db.fetchone(query, strict=True)
        return cls(imaging_depth=imaging_depth)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "ImagingDepth":
        return cls(imaging_depth=dict_repr['targeted_depth'])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "ImagingDepth":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(imaging_depth=metadata.imaging_depth)
