from pynwb import NWBFile

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from allensdk.brain_observatory.behavior.data_objects._base.readable_mixins\
    .lims_readable_mixin import \
    LimsReadableMixin
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_objects import DataObject


def from_lims_cache_key(cls, db, ophys_experiment_id: int):
    return hashkey(ophys_experiment_id)


class BehaviorSessionId(DataObject, LimsReadableMixin):
    def __init__(self, behavior_session_id: int):
        super().__init__(name="behavior_session_id", value=behavior_session_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "BehaviorSessionId":
        return cls(behavior_session_id=dict_repr["behavior_session_id"])

    def to_json(self) -> dict:
        return {"behavior_session_id": self.value}

    @classmethod
    @cached(cache=LRUCache(maxsize=10), key=from_lims_cache_key)
    def from_lims(
        cls, db: PostgresQueryMixin,
        ophys_experiment_id: int
    ) -> "BehaviorSessionId":
        query = f"""
            SELECT bs.id
            FROM ophys_experiments oe
            -- every ophys_experiment should have an ophys_session
            JOIN ophys_sessions os ON oe.ophys_session_id = os.id
            JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            WHERE oe.id = {ophys_experiment_id};
        """
        behavior_session_id = db.fetchone(query, strict=True)
        return cls(behavior_session_id=behavior_session_id)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "BehaviorSessionId":
        # TODO:
        # This implementation will be wrong when trying to get
        # behavior session id for an ophys experiment.
        # Will need to revisit.
        return cls(behavior_session_id=int(nwbfile.identifier))

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        # TODO:
        # Likely need to have some yet to be determined interaction with
        # the out-of-scope `metadata` DataObject
        raise NotImplementedError
