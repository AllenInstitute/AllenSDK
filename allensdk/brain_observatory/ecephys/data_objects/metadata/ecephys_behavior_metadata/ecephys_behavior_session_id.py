from pynwb import NWBFile

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    JsonWritableInterface
# from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_objects import DataObject


# def from_lims_cache_key(cls, db, ophys_experiment_id: int):
#     return hashkey(ophys_experiment_id)


class EcephysSessionId(
	DataObject,
    JsonReadableInterface,
    NwbReadableInterface,
    JsonWritableInterface
):
    def __init__(self, ecephys_session_id: int):
        super().__init__(name="ecephys_session_id", value=ecephys_session_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "EcephysSessionId":
        return cls(ecephys_session_id=dict_repr["ecephys_session_id"])

    def to_json(self) -> dict:
        return {"ecephys_session_id": self.value}

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "EcephysBehaviorSessionId":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(behavior_session_id=metadata.behavior_session_id)
