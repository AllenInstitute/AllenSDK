from typing import List, Dict

from pynwb import NWBFile

from allensdk.brain_observatory.behavior2.data_object import \
    DataObject


class BehaviorSessionId(DataObject):
    def __init__(self, behavior_session_id: int):
        super().__init__(name="behavior_session_id", value=behavior_session_id)

    @staticmethod
    def from_lims(dbconn) -> "BehaviorSessionId":
        raise NotImplementedError()

    @staticmethod
    def from_json(dict_repr: dict) -> "BehaviorSessionId":
        behavior_session_id = dict_repr['behavior_session_id']
        return BehaviorSessionId(behavior_session_id=behavior_session_id)

    @staticmethod
    def from_nwb(nwbfile: NWBFile) -> "BehaviorSessionId":
        behavior_session_id = int(nwbfile.identifier)
        return BehaviorSessionId(behavior_session_id=behavior_session_id)

    def to_json(self) -> Dict[str, int]:
        return {self._name: self._value}

    def to_nwb(self, nwbfile: NWBFile):
        raise NotImplementedError()
