from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_session_uuid import \
    BehaviorSessionUUID
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.stimulus_frame_rate import \
    StimulusFrameRate
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.subject_metadata import \
    SubjectMetadata
from allensdk.core import JsonReadableInterface, NwbReadableInterface


class BehaviorEcephysMetadata(BehaviorMetadata, JsonReadableInterface,
                              NwbReadableInterface):
    def __init__(
            self,
            ecephys_session_id: int,
            subject_metadata: SubjectMetadata,
            behavior_session_id: BehaviorSessionId,
            behavior_session_uuid: BehaviorSessionUUID,
            equipment: Equipment,
            session_type: SessionType,
            stimulus_frame_rate: StimulusFrameRate
    ):
        super().__init__(
            subject_metadata=subject_metadata,
            behavior_session_id=behavior_session_id,
            behavior_session_uuid=behavior_session_uuid,
            equipment=equipment,
            session_type=session_type,
            stimulus_frame_rate=stimulus_frame_rate
        )
        self._ecephys_session_id = ecephys_session_id

    @property
    def ecephys_session_id(self) -> int:
        return self._ecephys_session_id

    @classmethod
    def from_json(cls, dict_repr: dict) -> "BehaviorEcephysMetadata":
        behavior_metadata = super().from_json(dict_repr=dict_repr)
        return BehaviorEcephysMetadata(
            ecephys_session_id=dict_repr['ecephys_session_id'],
            subject_metadata=behavior_metadata.subject_metadata,
            behavior_session_id=behavior_metadata._behavior_session_id,
            behavior_session_uuid=behavior_metadata._behavior_session_uuid,
            equipment=behavior_metadata.equipment,
            session_type=behavior_metadata._session_type,
            stimulus_frame_rate=behavior_metadata._stimulus_frame_rate
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "BehaviorEcephysMetadata":
        behavior_metadata = super().from_nwb(nwbfile=nwbfile)
        return BehaviorEcephysMetadata(
            ecephys_session_id=int(nwbfile.identifier),
            behavior_session_id=behavior_metadata._behavior_session_id,
            behavior_session_uuid=behavior_metadata._behavior_session_uuid,
            equipment=behavior_metadata.equipment,
            session_type=behavior_metadata._session_type,
            stimulus_frame_rate=behavior_metadata._stimulus_frame_rate,
            subject_metadata=behavior_metadata.subject_metadata
        )
