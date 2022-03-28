import uuid
from typing import Dict, Optional
import re
import numpy as np
from pynwb import NWBFile

# from allensdk.brain_observatory.behavior.data_files import StimulusFile

from allensdk.brain_observatory.ecephys.data_objects.metadata \
    .ecephys_behavior_metadata.ecephys_behavior_session_id import \
    EcephysSessionId
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    JsonWritableInterface, NwbWritableInterface
from allensdk.brain_observatory.behavior.data_objects import DataObject

from allensdk.brain_observatory.ecephys.data_files \
    .ecephys_stimulus_file import \
    EcephysStimulusFile

from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_session_uuid import \
    BehaviorSessionUUID
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
# from allensdk.brain_observatory.behavior.data_objects.metadata\
#     .behavior_metadata.foraging_id import \
#     ForagingId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.stimulus_frame_rate import \
    StimulusFrameRate
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.subject_metadata import \
    SubjectMetadata
from allensdk.brain_observatory.behavior.schemas import BehaviorMetadataSchema
from allensdk.brain_observatory.nwb import load_pynwb_extension
# from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId

description_dict = {
    # key is a regex and value is returned on match
    r"\AOPHYS_0_images": "A behavior training session performed on the 2-photon calcium imaging setup but without recording neural activity, with the goal of habituating the mouse to the experimental setup before commencing imaging of neural activity. Habituation sessions are change detection with the same image set on which the mouse was trained. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_[1|3]_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse performs a visual change detection task with a set of natural images upon which it has been previously trained. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_2_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse is passively shown changes in natural scene images upon which it was previously trained as the change detection task is played in open loop mode, with the lick-response sensory withdrawn and the mouse is unable to respond to changes or receive reward feedback. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_[4|6]_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse performs a visual change detection task with natural scene images that are unique from those on which the mouse was trained prior to the imaging phase of the experiment. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\AOPHYS_5_images": "2-photon calcium imaging in the visual cortex of the mouse brain as the mouse is passively shown changes in natural scene images that are unique from those on which the mouse was trained prior to the imaging phase of the experiment. In this session, the change detection task is played in open loop mode, with the lick-response sensory withdrawn and the mouse is unable to respond to changes or receive reward feedback. Image stimuli are displayed for 250 ms with a 500 ms intervening gray period. 5% of non-change image presentations are randomly omitted. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session.",  # noqa: E501
    r"\ATRAINING_0_gratings": "An associative training session where a mouse is automatically rewarded when a grating stimulus changes orientation. Grating stimuli are  full-field, square-wave static gratings with a spatial frequency of 0.04 cycles per degree, with orientation changes between 0 and 90 degrees, at two spatial phases. Delivered rewards are 5ul in volume, and the session lasts for 15 minutes.",  # noqa: E501
    r"\ATRAINING_1_gratings": "An operant behavior training session where a mouse must lick following a change in stimulus identity to earn rewards. Stimuli consist of  full-field, square-wave static gratings with a spatial frequency of 0.04 cycles per degree. Orientation changes between 0 and 90 degrees occur with no intervening gray period. Delivered rewards are 10ul in volume, and the session lasts 60 minutes",  # noqa: E501
    r"\ATRAINING_2_gratings": "An operant behavior training session where a mouse must lick following a change in stimulus identity to earn rewards. Stimuli consist of full-field, square-wave static gratings with a spatial frequency of 0.04 cycles per degree. Gratings of 0 or 90 degrees are presented for 250 ms with a 500 ms intervening gray period. Delivered rewards are 10ul in volume, and the session lasts 60 minutes.",  # noqa: E501
    r"\ATRAINING_3_images": "An operant behavior training session where a mouse must lick following a change in stimulus identity to earn rewards. Stimuli consist of 8 natural scene images, for a total of 64 possible pairwise transitions. Images are shown for 250 ms with a 500 ms intervening gray period. Delivered rewards are 10ul in volume, and the session lasts for 60 minutes",  # noqa: E501
    r"\ATRAINING_4_images": "An operant behavior training session where a mouse must lick a spout following a change in stimulus identity to earn rewards. Stimuli consist of 8 natural scene images, for a total of 64 possible pairwise transitions. Images are shown for 250 ms with a 500 ms intervening gray period. Delivered rewards are 7ul in volume, and the session lasts for 60 minutes",  # noqa: E501
    r"\ATRAINING_5_images": "An operant behavior training session where a mouse must lick a spout following a change in stimulus identity to earn rewards. Stimuli consist of 8 natural scene images, for a total of 64 possible pairwise transitions. Images are shown for 250 ms with a 500 ms intervening gray period. Delivered rewards are 7ul in volume. The session is 75 minutes long, with 5 minutes of gray screen before and after 60 minutes of behavior, followed by 10 repeats of a 30 second natural movie stimulus at the end of the session."  # noqa: E501
    }


# def get_expt_description(session_type: str) -> str:
#     """Determine a behavior ophys session's experiment description based on
#     session type. Matches the regex patterns defined as the keys in
#     description_dict

#     Parameters
#     ----------
#     session_type : str
#         A session description string (e.g. OPHYS_1_images_B )

#     Returns
#     -------
#     str
#         A description of the experiment based on the session_type.

#     Raises
#     ------
#     RuntimeError
#         Behavior ophys sessions should only have 6 different session types.
#         Unknown session types (or malformed session_type strings) will raise
#         an error.
#     """
#     match = dict()
#     for k, v in description_dict.items():
#         if re.match(k, session_type) is not None:
#             match.update({k: v})

#     if len(match) != 1:
#         emsg = (f"session type should match one and only one possible pattern "
#                 f"template. '{session_type}' matched {len(match)} pattern "
#                 "templates.")
#         if len(match) > 1:
#             emsg += f"{list(match.keys())}"
#         emsg += f"the regex pattern templates are {list(description_dict)}"
#         raise RuntimeError(emsg)

#     return match.popitem()[1]


# def get_task_parameters(data: Dict) -> Dict:
#     """
#     Read task_parameters metadata from the behavior stimulus pickle file.

#     Parameters
#     ----------
#     data: dict
#         The nested dict read in from the behavior stimulus pickle file.
#         All of the data expected by this method lives under
#         data['items']['behavior']

#     Returns
#     -------
#     dict
#         A dict containing the task_parameters associated with this session.
#     """
#     behavior = data["items"]["behavior"]
#     stimuli = behavior['stimuli']
#     config = behavior["config"]
#     doc = config["DoC"]

#     task_parameters = {}

#     task_parameters['blank_duration_sec'] = \
#         [float(x) for x in doc['blank_duration_range']]

#     if 'images' in stimuli:
#         stim_key = 'images'
#     elif 'grating' in stimuli:
#         stim_key = 'grating'
#     else:
#         msg = "Cannot get stimulus_duration_sec\n"
#         msg += "'images' and/or 'grating' not a valid "
#         msg += "key in pickle file under "
#         msg += "['items']['behavior']['stimuli']\n"
#         msg += f"keys: {list(stimuli.keys())}"
#         raise RuntimeError(msg)

#     stim_duration = stimuli[stim_key]['flash_interval_sec']

#     # from discussion in
#     # https://github.com/AllenInstitute/AllenSDK/issues/1572
#     #
#     # 'flash_interval' contains (stimulus_duration, gray_screen_duration)
#     # (as @matchings said above). That second value is redundant with
#     # 'blank_duration_range'. I'm not sure what would happen if they were
#     # set to be conflicting values in the params. But it looks like
#     # they're always consistent. It should always be (0.25, 0.5),
#     # except for TRAINING_0 and TRAINING_1, which have statically
#     # displayed stimuli (no flashes).

#     if stim_duration is None:
#         stim_duration = np.NaN
#     else:
#         stim_duration = stim_duration[0]

#     task_parameters['stimulus_duration_sec'] = stim_duration

#     task_parameters['omitted_flash_fraction'] = \
#         behavior['params'].get('flash_omit_probability', float('nan'))
#     task_parameters['response_window_sec'] = \
#         [float(x) for x in doc["response_window"]]
#     task_parameters['reward_volume'] = config["reward"]["reward_volume"]
#     task_parameters['auto_reward_volume'] = doc['auto_reward_volume']
#     task_parameters['session_type'] = behavior["params"]["stage"]
#     task_parameters['stimulus'] = next(iter(behavior["stimuli"]))
#     task_parameters['stimulus_distribution'] = doc["change_time_dist"]

#     task_id = config['behavior']['task_id']
#     if 'DoC' in task_id:
#         task_parameters['task'] = 'change detection'
#     else:
#         msg = "metadata.get_task_parameters does not "
#         msg += f"know how to parse 'task_id' = {task_id}"
#         raise RuntimeError(msg)

#     n_stimulus_frames = 0
#     for stim_type, stim_table in behavior["stimuli"].items():
#         n_stimulus_frames += sum(stim_table.get("draw_log", []))
#     task_parameters['n_stimulus_frames'] = n_stimulus_frames

#     return task_parameters


class EcephysBehaviorMetadata(DataObject,
                       JsonReadableInterface,
                       NwbReadableInterface,
                       JsonWritableInterface,
                       NwbWritableInterface):
    # """Container class for behavior metadata"""
    # def __init__(self,
    #              subject_metadata: SubjectMetadata,
    #              behavior_session_id: BehaviorSessionId,
    #              equipment: Equipment,
    #              stimulus_frame_rate: StimulusFrameRate,
    #              session_type: SessionType,
    #              behavior_session_uuid: BehaviorSessionUUID):
    #     super().__init__(name='behavior_metadata', value=self)
    #     self._subject_metadata = subject_metadata
    #     self._behavior_session_id = behavior_session_id
    #     self._equipment = equipment
    #     self._stimulus_frame_rate = stimulus_frame_rate
    #     self._session_type = session_type
    #     self._behavior_session_uuid = behavior_session_uuid

    #     self._exclude_from_equals = set()

    def __init__(self,
        subject_metadata: SubjectMetadata,
        ecephys_session_id: EcephysSessionId,
        behavior_stimulus_file: EcephysStimulusFile,
        mapping_stimulus_file: EcephysStimulusFile,
        replay_stimulus_file: EcephysStimulusFile,
        equipment: Equipment,
        stimulus_frame_rate: StimulusFrameRate,
        session_type: SessionType,
        behavior_session_uuid: BehaviorSessionUUID,
        behavior_session_id: BehaviorSessionId

    ):
        self._subject_metadata = subject_metadata
        self._ecephys_session_id = ecephys_session_id
        self._behavior_stimulus_file = behavior_stimulus_file
        self._mapping_stimulus_file = mapping_stimulus_file
        self._replay_stimulus_file = replay_stimulus_file
        self._equipment = equipment
        self._stimulus_frame_rate = stimulus_frame_rate
        self._session_type = session_type
        self._behavior_session_uuid = behavior_session_uuid
        self._behavior_session_id = behavior_session_id
        super().__init__(name='ecephys_behavior_metadata', value=self)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "EcephysBehaviorMetadata":
        subject_metadata = SubjectMetadata.from_json(dict_repr=dict_repr)
        ecephys_session_id = EcephysSessionId.from_json(dict_repr=dict_repr)
        behavior_session_id = BehaviorSessionId.from_json(dict_repr=dict_repr)

        behavior_stimulus_file = EcephysStimulusFile.from_json(dict_repr=dict_repr, file_type='behavior_pkl_path')
        mapping_stimulus_file = EcephysStimulusFile.from_json(dict_repr=dict_repr, file_type='mapping_pkl_path')
        replay_stimulus_file = EcephysStimulusFile.from_json(dict_repr=dict_repr, file_type='replay_pkl_path')

        equipment = Equipment.from_json(dict_repr=dict_repr)

        stimulus_frame_rate = StimulusFrameRate.from_stimulus_file(
            stimulus_file=behavior_stimulus_file)
        session_type = SessionType.from_stimulus_file(
            stimulus_file=behavior_stimulus_file)
        session_uuid = BehaviorSessionUUID.from_stimulus_file(
            stimulus_file=behavior_stimulus_file)

        # return cls(
        #     subject_metadata=subject_metadata,
        #     behavior_session_id=behavior_session_id,
        #     equipment=equipment,
        #     stimulus_frame_rate=stimulus_frame_rate,
        #     session_type=session_type,
        #     behavior_session_uuid=session_uuid,
        # )
        return cls(
            subject_metadata=subject_metadata,
            ecephys_session_id=ecephys_session_id,
            behavior_session_id=behavior_session_id,
            behavior_stimulus_file=behavior_stimulus_file,
            mapping_stimulus_file=mapping_stimulus_file,
            replay_stimulus_file=replay_stimulus_file,
            equipment=equipment,
            stimulus_frame_rate=stimulus_frame_rate,
            session_type=session_type,
            behavior_session_uuid=session_uuid
        )

  # "ecephys_session_id": 1101473342,
  #   "foraging_id": "767641ef119d4db6ad833c0280dc563d",
  #   "driver_line": [
  #     "Sst-IRES-Cre"
  #   ],
  #   "reporter_line": [
  #     "Ai32(RCL-ChR2(H134R)_EYFP)"
  #   ],
  #   "full_genotype": "Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt",
  #   "rig_name": "NP.1",
  #   "date_of_acquisition": "2021-05-06 14:01:47",
  #   "external_specimen_name": 564012,
  #   "sync_h5_path": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_1080503595/ecephys_session_1101473342/1101586032/1101473342_564012_20210506.sync",
  #   "behavior_pkl_path": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_1080503595/ecephys_session_1101473342/1101586032/1101473342_564012_20210506.behavior.pkl",
  #   "mapping_pkl_path": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_1080503595/ecephys_session_1101473342/1101586032/1101473342_564012_20210506.mapping.pkl",
  #   "replay_pkl_path": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_1080503595/ecephys_session_1101473342/1101586032/1101473342_564012_20210506.replay.pkl",
  #   "stim_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_1080503595/ecephys_session_1101473342/VbnCreateStimTableStrategy/analysis_run_1151227584/1101473342_VBN_stimulus_table.csv"

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "BehaviorMetadata":
        subject_metadata = SubjectMetadata.from_nwb(nwbfile=nwbfile)

        behavior_session_id = BehaviorSessionId.from_nwb(nwbfile=nwbfile)
        equipment = Equipment.from_nwb(nwbfile=nwbfile)
        stimulus_frame_rate = StimulusFrameRate.from_nwb(nwbfile=nwbfile)
        session_type = SessionType.from_nwb(nwbfile=nwbfile)
        session_uuid = BehaviorSessionUUID.from_nwb(nwbfile=nwbfile)

        return cls(
            subject_metadata=subject_metadata,
            behavior_session_id=behavior_session_id,
            equipment=equipment,
            stimulus_frame_rate=stimulus_frame_rate,
            session_type=session_type,
            behavior_session_uuid=session_uuid
        )

    @property
    def equipment(self) -> Equipment:
        return self._equipment

    @property
    def stimulus_frame_rate(self) -> float:
        return self._stimulus_frame_rate.value

    @property
    def session_type(self) -> str:
        return self._session_type.value

    @property
    def behavior_session_uuid(self) -> Optional[uuid.UUID]:
        return self._behavior_session_uuid.value

    @property
    def ecephys_session_id(self) -> int:
        return self._ecephys_session_id.value

    @property
    def subject_metadata(self):
        return self._subject_metadata

    @property
    def behavior_session_id(self):
        return self._behavior_session_id.value

    def to_json(self) -> dict:
        pass

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        try:
            self._subject_metadata.to_nwb(nwbfile=nwbfile)
        except Exception as error:
            pass

        self._equipment.to_nwb(nwbfile=nwbfile)
        extension = load_pynwb_extension(BehaviorMetadataSchema,
                                                'ndx-aibs-behavior-ophys')
        nwb_metadata = extension(
            name='metadata',
            behavior_session_id=self.behavior_session_id,
            behavior_session_uuid=str(self.behavior_session_uuid),
            stimulus_frame_rate=self.stimulus_frame_rate,
            session_type=self.session_type,
            equipment_name=self.equipment.value
        )
        nwbfile.add_lab_meta_data(nwb_metadata)

        return nwbfile
