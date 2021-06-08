import uuid

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects._base\
    .readable_interfaces.json_readable_interface import \
    JsonReadableInterface
from allensdk.brain_observatory.behavior.data_objects._base.readable_interfaces\
    .lims_readable_interface import \
    LimsReadableInterface
from allensdk.brain_observatory.behavior.data_objects._base\
    .writable_interfaces.json_writable_interface import \
    JsonWritableInterface
from allensdk.internal.api import PostgresQueryMixin


class ForagingId(DataObject, LimsReadableInterface, JsonReadableInterface,
                 JsonWritableInterface):
    """Foraging id"""
    def __init__(self, foraging_id: uuid.UUID):
        super().__init__(name="foraging_id", value=foraging_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "ForagingId":
        pass

    def to_json(self) -> dict:
        return {"sex": self.value}

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: PostgresQueryMixin) -> "ForagingId":
        query = f"""
            SELECT
                foraging_id
            FROM
                behavior_sessions
            WHERE
                behavior_sessions.id = {behavior_session_id};
        """
        foraging_id = lims_db.fetchone(query, strict=True)
        foraging_id = uuid.UUID(foraging_id)
        return cls(foraging_id=foraging_id)
