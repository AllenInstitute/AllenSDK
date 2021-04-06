from typing import List, Dict
from allensdk.brain_observatory.behavior2.data_object import \
    DataObject


class OphysExperimentIds(DataObject):
    def __init__(self, ophys_experiment_ids: List[int]):
        super().__init__(name="ophys_experiment_ids",
                         value=ophys_experiment_ids)

    @staticmethod
    def from_lims(dbconn, ophys_session_id: int) -> "OphysExperimentIds":
        dbresponse = dbconn.fetchall(
                f"""
                SELECT id
                FROM ophys_experiments
                WHERE ophys_session_id = {ophys_session_id}
                """)
        if not len(dbresponse):
            value = None
        else:
            value = [int(i) for i in dbresponse]
        return OphysExperimentIds(ophys_experiment_ids=value)

    @staticmethod
    def from_json(dict_repr: dict) -> "OphysExperimentIds":
        ophys_experiment_ids = dict_repr['ophys_experiment_ids']
        return OphysExperimentIds(ophys_experiment_ids=ophys_experiment_ids)

    def from_nwb(self):
        pass

    def to_nwb(self):
        pass

    def to_json(self) -> Dict[str, int]:
        return {self._name: self._value}


class OphysSessionId(DataObject):
    def __init__(self, ophys_session_id: int):
        super().__init__(name="ophys_session_id", value=ophys_session_id)

    @staticmethod
    def from_lims(dbconn, behavior_session_id) -> "OphysSessionId":
        dbresponse = dbconn.fetchall(
                f"""
                SELECT os.id
                FROM ophys_sessions os
                JOIN behavior_sessions bs
                ON os.id = bs.ophys_session_id
                WHERE bs.id = {behavior_session_id}
                """)
        if not len(dbresponse):
            value = None
        else:
            value = int(dbresponse[0])
        return OphysSessionId(ophys_session_id=value)

    @staticmethod
    def from_json(dict_rep: dict) -> "OphysSessionId":
        ophys_session_id = dict_rep['ophys_session_id']
        return OphysSessionId(ophys_session_id=ophys_session_id)

    def from_nwb(self):
        pass

    def to_nwb(self):
        pass

    def to_json(self) -> Dict[str, int]:
        return {self._name: self._value}


class BehaviorSessionId(DataObject):
    def __init__(self, behavior_session_id: int):
        super().__init__(name="behavior_session_id", value=behavior_session_id)

    @staticmethod
    def from_lims(dbconn, ophys_experiment_id) -> "BehaviorSessionId":
        dbresponse = dbconn.fetchall(
                f"""
                SELECT bs.id
                FROM ophys_experiments oe
                JOIN ophys_sessions os
                ON oe.ophys_session_id = os.id
                JOIN behavior_sessions bs
                ON os.id = bs.ophys_session_id
                WHERE oe.id = {ophys_experiment_id}
                """)
        if not len(dbresponse):
            value = None
        else:
            value = int(dbresponse[0])
        return BehaviorSessionId(behavior_session_id=value)

    @staticmethod
    def from_json(dict_repr: dict) -> "BehaviorSessionId":
        behavior_session_id = dict_repr['behavior_session_id']
        return BehaviorSessionId(behavior_session_id=behavior_session_id)

    def from_nwb(self):
        pass

    def to_json(self) -> Dict[str, int]:
        return {self._name: self._value}

    def to_nwb(self):
        pass
