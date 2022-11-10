from pynwb import NWBFile

from allensdk.core import DataObject, JsonReadableInterface, LimsReadableInterface, NwbReadableInterface  # NOQA
from allensdk.internal.api import PostgresQueryMixin


class TargetImagingDepth(
    DataObject,
    LimsReadableInterface,
    NwbReadableInterface,
    JsonReadableInterface,
):
    def __init__(self, target_imaging_depth: int):
        super().__init__(
            name="target_imaging_depth", value=target_imaging_depth
        )

    @classmethod
    def from_lims(
        cls, ophys_experiment_id: int, lims_db: PostgresQueryMixin
    ) -> "TargetImagingDepth":
        query_container_id = """
            SELECT visual_behavior_experiment_container_id
            FROM ophys_experiments_visual_behavior_experiment_containers
            WHERE ophys_experiment_id = {}
        """.format(
            ophys_experiment_id
        )

        container_id = lims_db.fetchone(query_container_id, strict=True)

        query_depths = """
            SELECT id.depth
            FROM ophys_experiments_visual_behavior_experiment_containers ec
            JOIN ophys_experiments oe ON oe.id = ec.ophys_experiment_id
            JOIN ophys_sessions os ON oe.ophys_session_id = os.id
            LEFT JOIN imaging_depths id ON id.id = oe.imaging_depth_id
            WHERE ec.visual_behavior_experiment_container_id = {};
        """.format(
            container_id
        )

        imaging_depths = lims_db.fetchall(query_depths)
        target_imaging_depth = round(sum(imaging_depths) / len(imaging_depths))
        return cls(target_imaging_depth=target_imaging_depth)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "TargetImagingDepth":
        return cls(target_imaging_depth=dict_repr["targeted_imaging_depth"])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "TargetImagingDepth":
        metadata = nwbfile.lab_meta_data["metadata"]
        return cls(target_imaging_depth=metadata.target_imaging_depth)
