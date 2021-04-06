from pynwb import NWBFile, NWBHDF5IO

from allensdk.brain_observatory.behavior2.data_object import \
    DataObject
from allensdk.brain_observatory.behavior2.data_objects.ids import \
    BehaviorSessionId, OphysExperimentIds, OphysSessionId
from allensdk.brain_observatory.behavior2.data_objects.running_speed\
    .running_speed import \
    RunningSpeed
from allensdk.brain_observatory.behavior2.data_objects.stimulus_timestamps import \
    StimulusTimestamps
from allensdk.brain_observatory.behavior2.stimulus_file import StimulusFile


class BehaviorSession(DataObject):
    def __init__(self,
                 behavior_session_id: BehaviorSessionId,
                 ophys_session_id: OphysSessionId,
                 ophys_experiment_ids: OphysExperimentIds,
                 stimulus_timestamps: StimulusTimestamps,
                 running_speed: RunningSpeed):
        self._behavior_session_id = behavior_session_id
        self._ophys_session_id = ophys_session_id
        self._ophys_experiment_ids = ophys_experiment_ids
        self._stimulus_timestamps = stimulus_timestamps
        self._running_speed = running_speed

        super().__init__(name='BehaviorSession', value=None)

    @property
    def behavior_session_id(self):
        return self._behavior_session_id

    @property
    def ophys_session_id(self):
        return self._ophys_session_id

    @property
    def ophys_experiment_ids(self):
        return self._ophys_experiment_ids

    @property
    def stimulus_timestamps(self):
        return self._stimulus_timestamps

    @property
    def running_speed(self):
        return self._running_speed

    @staticmethod
    def from_lims(dbconn, ophys_experiment_id) -> "BehaviorSession":
        behavior_session_id = BehaviorSessionId.from_lims(
                dbconn, ophys_experiment_id)
        ophys_session_id = OphysSessionId.from_lims(
                dbconn, behavior_session_id.value)
        ophys_experiment_ids = OphysExperimentIds.from_lims(
                dbconn, ophys_session_id.value)
        stimulus_file = StimulusFile.from_lims(
            dbconn=dbconn, behavior_session_id=behavior_session_id.value)
        stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
            stimulus_file=stimulus_file)
        running_speed = RunningSpeed.from_stimulus_file(
            stimulus_file=stimulus_file)
        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            ophys_session_id=ophys_session_id,
            ophys_experiment_ids=ophys_experiment_ids,
            stimulus_timestamps=stimulus_timestamps,
            running_speed=running_speed)

    @staticmethod
    def from_json(dict_repr) -> "BehaviorSession":
        behavior_session_id = BehaviorSessionId.from_json(dict_repr=dict_repr)
        ophys_session_id = OphysSessionId.from_json(dict_rep=dict_repr)
        ophys_experiment_ids = \
            OphysExperimentIds.from_json(dict_repr=dict_repr)
        stimulus_timestamps = StimulusTimestamps.from_json(dict_repr=dict_repr)
        running_speed = RunningSpeed.from_json(dict_repr=dict_repr)
        return BehaviorSession(
            behavior_session_id=behavior_session_id,
            ophys_session_id=ophys_session_id,
            ophys_experiment_ids=ophys_experiment_ids,
            stimulus_timestamps=stimulus_timestamps,
            running_speed=running_speed)

    def from_nwb(self):
        pass

    def to_json(self):
        dict_repr = dict()
        for prop, value in vars(self).items():
            if (prop.startswith('_') &
                    isinstance(value, DataObject)):
                dict_repr.update(value.to_json())
        return dict_repr

    def write_nwb(self, path: str, nwbfile: NWBFile):
        self.to_nwb(nwbfile=nwbfile)
        with NWBHDF5IO(path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

    def to_nwb(self, nwbfile: NWBFile):
        self._stimulus_timestamps.to_nwb(nwbfile=nwbfile)
        self._running_speed.to_nwb(nwbfile=nwbfile)
