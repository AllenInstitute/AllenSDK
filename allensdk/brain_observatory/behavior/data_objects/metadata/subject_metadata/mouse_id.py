from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class MouseId(DataObject, LimsReadableInterface, JsonReadableInterface,
              NwbReadableInterface):
    """the LabTracks ID"""
    def __init__(self, mouse_id: int):
        super().__init__(name="mouse_id", value=mouse_id)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "MouseId":
        mouse_id = dict_repr['external_specimen_name']
        mouse_id = int(mouse_id)
        return cls(mouse_id=mouse_id)

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: PostgresQueryMixin) -> "MouseId":
        # TODO: Should this even be included?
        # Found sometimes there were entries with NONE which is
        # why they are filtered out; also many entries in the table
        # match the donor_id, which is why used DISTINCT
        query = f"""
            SELECT DISTINCT(sp.external_specimen_name)
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN specimens sp ON sp.donor_id=d.id
            WHERE bs.id={behavior_session_id}
            AND sp.external_specimen_name IS NOT NULL;
            """
        mouse_id = int(lims_db.fetchone(query, strict=True))
        return cls(mouse_id=mouse_id)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "MouseId":
        return cls(mouse_id=int(nwbfile.subject.subject_id))
