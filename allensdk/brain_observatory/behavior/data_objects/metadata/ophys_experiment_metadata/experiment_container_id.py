from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin
from allensdk import OneResultExpectedError


class ExperimentContainerId(DataObject, LimsReadableInterface,
                            JsonReadableInterface, NwbReadableInterface):
    """"experiment container id"""
    def __init__(self, experiment_container_id: int):
        super().__init__(name='experiment_container_id',
                         value=experiment_container_id)

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin) -> "ExperimentContainerId":
        query = """
                SELECT visual_behavior_experiment_container_id
                FROM ophys_experiments_visual_behavior_experiment_containers
                WHERE ophys_experiment_id = {};
                """.format(ophys_experiment_id)
        try:
            container_id = lims_db.fetchone(query, strict=False)
        except OneResultExpectedError:
            print("No single container id found for {}!".format(ophys_experiment_id))
            container_id = lims_db.fetchall(query, strict=False)
            print("Container id query returned {}".format(container_id))
        return cls(experiment_container_id=container_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "ExperimentContainerId":
        return cls(experiment_container_id=dict_repr['container_id'])

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "ExperimentContainerId":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(experiment_container_id=metadata.experiment_container_id)
