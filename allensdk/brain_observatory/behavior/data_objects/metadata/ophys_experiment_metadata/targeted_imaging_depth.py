from pynwb import NWBFile

from allensdk.core import DataObject, JsonReadableInterface, LimsReadableInterface, NwbReadableInterface  # NOQA
from allensdk.internal.api import PostgresQueryMixin


class TargetedImagingDepth(
    DataObject,
    LimsReadableInterface,
    NwbReadableInterface,
    JsonReadableInterface,
):
    """Data object loads and stores the average `imaging_depth`s
    (microns) across experiments in the container that an experiment is
    associated with.
    """
    def __init__(self, targeted_imaging_depth: int):
        super().__init__(
            name="targeted_imaging_depth", value=targeted_imaging_depth
        )

    @classmethod
    def from_lims(
        cls, ophys_experiment_id: int, lims_db: PostgresQueryMixin
    ) -> "TargetedImagingDepth":
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

        targeted_imaging_depth = int(lims_db.fetchone(query_depths))
        return cls(targeted_imaging_depth=targeted_imaging_depth)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "TargetedImagingDepth":
        # TODO remove all of the from_json loading and validation step
        # ticket 2607
        return cls(targeted_imaging_depth=dict_repr["targeted_depth"])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "TargetedImagingDepth":
        try:
            metadata = nwbfile.lab_meta_data["metadata"]
            return cls(targeted_imaging_depth=metadata.targeted_imaging_depth)
        except AttributeError:
            return None
