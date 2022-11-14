from pynwb import NWBFile

from allensdk.core import DataObject, JsonReadableInterface, LimsReadableInterface, NwbReadableInterface  # NOQA
from allensdk.internal.api import PostgresQueryMixin


class AverageContainerDepth(
    DataObject,
    LimsReadableInterface,
    NwbReadableInterface,
    JsonReadableInterface,
):
    def __init__(self, average_container_depth: int):
        super().__init__(
            name="average_container_depth", value=average_container_depth
        )

    @classmethod
    def from_lims(
        cls, ophys_experiment_id: int, lims_db: PostgresQueryMixin
    ) -> "AverageContainerDepth":
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
        average_container_depth = round(sum(imaging_depths) /
                                        len(imaging_depths))
        return cls(average_container_depth=average_container_depth)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "AverageContainerDepth":
        return cls(average_container_depth=dict_repr["targeted_imaging_depth"])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "AverageContainerDepth":
        metadata = nwbfile.lab_meta_data["metadata"]
        return cls(average_container_depth=metadata.average_container_depth)
