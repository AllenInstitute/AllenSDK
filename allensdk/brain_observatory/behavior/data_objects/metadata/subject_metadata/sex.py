from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base\
    .writable_interfaces import \
    JsonWritableInterface
from allensdk.internal.api import PostgresQueryMixin


class Sex(DataObject, LimsReadableInterface, JsonReadableInterface,
          NwbReadableInterface, JsonWritableInterface):
    """sex of the animal (M/F)"""
    def __init__(self, sex: str):
        super().__init__(name="sex", value=sex)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "Sex":
        return cls(sex=dict_repr["sex"])

    def to_json(self) -> dict:
        return {"sex": self.value}

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: PostgresQueryMixin) -> "Sex":
        query = f"""
            SELECT g.name AS sex
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id = d.id
            JOIN genders g ON g.id = d.gender_id
            WHERE bs.id = {behavior_session_id};
            """
        sex = lims_db.fetchone(query, strict=True)
        return cls(sex=sex)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "Sex":
        return cls(sex=nwbfile.subject.sex)
