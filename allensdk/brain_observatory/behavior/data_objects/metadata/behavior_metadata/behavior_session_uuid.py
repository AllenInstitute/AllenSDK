import uuid
from typing import Optional

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    NwbReadableInterface, StimulusFileReadableInterface


class BehaviorSessionUUID(DataObject, StimulusFileReadableInterface,
                          NwbReadableInterface):
    """the universally unique identifier (UUID)"""
    def __init__(self, behavior_session_uuid: Optional[uuid.UUID]):
        super().__init__(name="behavior_session_uuid",
                         value=behavior_session_uuid)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile) -> "BehaviorSessionUUID":
        id = stimulus_file.data.get('session_uuid')
        if id:
            id = uuid.UUID(id)
        return cls(behavior_session_uuid=id)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "BehaviorSessionUUID":
        metadata = nwbfile.lab_meta_data['metadata']
        id = uuid.UUID(metadata.behavior_session_uuid)
        return cls(behavior_session_uuid=id)

    def validate(self, behavior_session_id: int,
                 foraging_id: int,
                 stimulus_file: StimulusFile) -> "BehaviorSessionUUID":
        """
        Sanity check to ensure that pkl file data matches up with
        the behavior session that the pkl file has been associated with.
        """
        assert_err_msg = (
            f"The behavior session UUID ({self.value}) in the "
            f"behavior stimulus *.pkl file "
            f"({stimulus_file.filepath}) does "
            f"does not match the foraging UUID ({foraging_id}) for "
            f"behavior session: {behavior_session_id}")
        assert self.value == foraging_id, assert_err_msg

        return self
