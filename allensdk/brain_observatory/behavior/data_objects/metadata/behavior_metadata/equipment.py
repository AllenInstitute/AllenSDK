from enum import Enum

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    JsonWritableInterface, NwbWritableInterface
from allensdk.internal.api import PostgresQueryMixin


class EquipmentType(Enum):
    MESOSCOPE = 'MESOSCOPE'
    OTHER = 'OTHER'


class Equipment(DataObject, JsonReadableInterface, LimsReadableInterface,
                NwbReadableInterface, JsonWritableInterface,
                NwbWritableInterface):
    """the name of the experimental rig."""
    def __init__(self, equipment_name: str):
        super().__init__(name="equipment_name", value=equipment_name)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "Equipment":
        return cls(equipment_name=dict_repr["rig_name"])

    def to_json(self) -> dict:
        return {"eqipment_name": self.value}

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: PostgresQueryMixin) -> "Equipment":
        query = f"""
            SELECT e.name AS device_name
            FROM behavior_sessions bs
            JOIN equipment e ON e.id = bs.equipment_id
            WHERE bs.id = {behavior_session_id};
        """
        equipment_name = lims_db.fetchone(query, strict=True)
        return cls(equipment_name=equipment_name)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "Equipment":
        metadata = nwbfile.lab_meta_data['metadata']
        return cls(equipment_name=metadata.equipment_name)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        if self.type == EquipmentType.MESOSCOPE:
            device_config = {
                "name": self.value,
                "description": "Allen Brain Observatory - Mesoscope 2P Rig"
            }
        else:
            device_config = {
                "name": self.value,
                "description": "Allen Brain Observatory - Scientifica 2P "
                               "Rig",
                "manufacturer": "Scientifica"
            }
        nwbfile.create_device(**device_config)
        return nwbfile

    @property
    def type(self):
        if self.value.startswith('MESO'):
            et = EquipmentType.MESOSCOPE
        else:
            et = EquipmentType.OTHER
        return et
