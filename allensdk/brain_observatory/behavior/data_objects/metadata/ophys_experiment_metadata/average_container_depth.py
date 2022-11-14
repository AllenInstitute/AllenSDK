from pynwb import NWBFile

from allensdk.core import DataObject, JsonReadableInterface, LimsReadableInterface, NwbReadableInterface  # NOQA
from allensdk.internal.api import PostgresQueryMixin


class AverageContainerDepth(
    DataObject,
    LimsReadableInterface,
    NwbReadableInterface,
    JsonReadableInterface,
):
    """Data object loads and stores the average `imaging_depth`s
    (microns) across experiments in the container that an experiment is
    associated with.
    """
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
            SELECT AVG(imd.depth)
            FROM ophys_experiments_visual_behavior_experiment_containers ec
            JOIN ophys_experiments oe ON oe.id = ec.ophys_experiment_id
            LEFT JOIN imaging_depths imd ON imd.id = oe.imaging_depth_id
            WHERE ec.visual_behavior_experiment_container_id = {};
        """.format(
            container_id
        )

        average_container_depth = int(lims_db.fetchone(query_depths))
        return cls(average_container_depth=average_container_depth)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "AverageContainerDepth":
        return cls(average_container_depth=dict_repr["targeted_imaging_depth"])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "AverageContainerDepth":
        metadata = nwbfile.lab_meta_data["metadata"]
        return cls(average_container_depth=metadata.average_container_depth)
