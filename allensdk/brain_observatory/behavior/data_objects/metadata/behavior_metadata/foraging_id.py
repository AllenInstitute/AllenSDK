import uuid

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class ForagingId(DataObject, LimsReadableInterface, JsonReadableInterface):
    """Foraging id"""
    def __init__(self, foraging_id: uuid.UUID):
        super().__init__(name="foraging_id", value=foraging_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "ForagingId":
        pass

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
