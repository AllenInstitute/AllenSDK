from allensdk.brain_observatory.behavior2.data_objects.abc import \
    AbstractDataObject
from allensdk.brain_observatory.behavior2.data_objects.ids import \
    BehaviorSessionId, OphysExperimentIds, OphysSessionId


class BehaviorSession(AbstractDataObject):
    def __init__(self,
                 behavior_session_id: BehaviorSessionId,
                 ophys_session_id: OphysSessionId,
                 ophys_experiment_ids: OphysExperimentIds):
        self._behavior_session_id = behavior_session_id
        self._ophys_session_id = ophys_session_id
        self._ophys_experiment_ids = ophys_experiment_ids
        self.set_properties()

    @staticmethod
    def from_lims(dbconn, ophys_experiment_id):
        behavior_session_id = BehaviorSessionId.from_lims(
                dbconn, ophys_experiment_id)
        ophys_session_id = OphysSessionId.from_lims(
                dbconn, behavior_session_id._value)
        ophys_experiment_ids = OphysExperimentIds.from_lims(
                dbconn, ophys_session_id._value)
        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            ophys_session_id=ophys_session_id,
            ophys_experiment_ids=ophys_experiment_ids)

    @staticmethod
    def from_dict(dict_repr):
        behavior_session_id = \
            BehaviorSessionId(dict_repr["behavior_session_id"])
        ophys_session_id = \
            OphysSessionId(dict_repr["ophys_session_id"])
        ophys_experiment_ids = \
            OphysExperimentIds(dict_repr["ophys_experiment_ids"])
        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            ophys_session_id=ophys_session_id,
            ophys_experiment_ids=ophys_experiment_ids)

    def set_properties(self):
        freeze_vars = list(vars(self).items())
        for property, value in freeze_vars:
            if (property.startswith('_') &
                    isinstance(value, AbstractDataObject)):
                setattr(self, property[1:], value._value)

    def to_dict(self):
        dict_repr = dict()
        for property, value in vars(self).items():
            if (property.startswith('_') &
                    isinstance(value, AbstractDataObject)):
                dict_repr.update(value.to_dict())
        return dict_repr
