from pynwb import NWBFile

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    JsonWritableInterface
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_objects import DataObject


def from_lims_cache_key(cls, db, ophys_experiment_id: int):
    return hashkey(ophys_experiment_id)


class VisualCodingSessionId(DataObject, LimsReadableInterface,
                        JsonReadableInterface,
                        NwbReadableInterface,
                        JsonWritableInterface):
    def __init__(self, ophys_session_id: int):
        super().__init__(name="ophys_session_id", value=ophys_session_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "VisualCodingSessionId":
        return cls(ophys_session_id=dict_repr["ophys_session_id"])

    def to_json(self) -> dict:
        return {"ophys_session_id": self.value}

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin,
        ophys_experiment_id: int
    ) -> "VisualCodingSessionId":
        query = f"""
            SELECT oe.ophys_session_id
            FROM ophys_experiments oe
            -- every ophys_experiment should have an ophys_session
            WHERE oe.id = {ophys_experiment_id};
        """
        ophys_session_id = db.fetchone(query, strict=True)
        return cls(ophys_session_id=ophys_session_id)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "VisualCodingSessionId":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(ophys_session_id=metadata.ophys_session_id)
