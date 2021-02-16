from typing import List, Dict
from allensdk.brain_observatory.behavior2.data_objects.abc import \
    AbstractDataObject


class OphysExperimentIds(AbstractDataObject):
    def __init__(self, ophys_experiment_ids: List[int]):
        self._name = "ophys_experiment_ids"
        self._value = ophys_experiment_ids

    def to_dict(self) -> Dict[str, int]:
        return {self._name: self._value}

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


class OphysSessionId(AbstractDataObject):
    def __init__(self, ophys_session_id: int):
        self._name = "ophys_session_id"
        self._value = ophys_session_id

    def to_dict(self) -> Dict[str, int]:
        return {self._name: self._value}

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


class BehaviorSessionId(AbstractDataObject):
    def __init__(self, behavior_session_id: int):
        self._name = "behavior_session_id"
        self._value = behavior_session_id

    def to_dict(self) -> Dict[str, int]:
        return {self._name: self._value}

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
