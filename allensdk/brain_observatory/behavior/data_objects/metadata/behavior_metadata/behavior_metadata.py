import abc
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import re
import numpy as np
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps, BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects._base.readable_mixins\
    .internal_mixed_readable_mixin \
    import \
    InternalMixedReadableMixin
from allensdk.brain_observatory.behavior.data_objects._base.writable_mixins\
    .nwb_writable_mixin import \
    NwbWritableMixin
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.age import \
    Age
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_session_uuid import \
    BehaviorSessionUUID
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.driver_line import \
    DriverLine
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment_name import \
    EquipmentName
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.foraging_id import \
    ForagingId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.full_genotype import \
    FullGenotype
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.mouse_id import \
    MouseId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.reporter_line import \
    ReporterLine
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.sex import \
    Sex
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.stimulus_frame_rate import \
    StimulusFrameRate
from allensdk.brain_observatory.behavior.schemas import SubjectMetadataSchema, \
    CompleteOphysBehaviorMetadataSchema, BehaviorMetadataSchema
from allensdk.brain_observatory.nwb import load_pynwb_extension
from allensdk.brain_observatory.session_api_utils import compare_session_fields
from allensdk.internal.api import PostgresQueryMixin

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


def get_expt_description(session_type: str) -> str:
    """Determine a behavior ophys session's experiment description based on
    session type. Matches the regex patterns defined as the keys in
    description_dict

    Parameters
    ----------
    session_type : str
        A session description string (e.g. OPHYS_1_images_B )

    Returns
    -------
    str
        A description of the experiment based on the session_type.

    Raises
    ------
    RuntimeError
        Behavior ophys sessions should only have 6 different session types.
        Unknown session types (or malformed session_type strings) will raise
        an error.
    """
    match = dict()
    for k, v in description_dict.items():
        if re.match(k, session_type) is not None:
            match.update({k: v})

    if len(match) != 1:
        emsg = (f"session type should match one and only one possible pattern "
                f"template. '{session_type}' matched {len(match)} pattern "
                "templates.")
        if len(match) > 1:
            emsg += f"{list(match.keys())}"
        emsg += f"the regex pattern templates are {list(description_dict)}"
        raise RuntimeError(emsg)

    return match.popitem()[1]


def get_task_parameters(data: Dict) -> Dict:
    """
    Read task_parameters metadata from the behavior stimulus pickle file.

    Parameters
    ----------
    data: dict
        The nested dict read in from the behavior stimulus pickle file.
        All of the data expected by this method lives under
        data['items']['behavior']

    Returns
    -------
    dict
        A dict containing the task_parameters associated with this session.
    """
    behavior = data["items"]["behavior"]
    stimuli = behavior['stimuli']
    config = behavior["config"]
    doc = config["DoC"]

    task_parameters = {}

    task_parameters['blank_duration_sec'] = \
        [float(x) for x in doc['blank_duration_range']]

    if 'images' in stimuli:
        stim_key = 'images'
    elif 'grating' in stimuli:
        stim_key = 'grating'
    else:
        msg = "Cannot get stimulus_duration_sec\n"
        msg += "'images' and/or 'grating' not a valid "
        msg += "key in pickle file under "
        msg += "['items']['behavior']['stimuli']\n"
        msg += f"keys: {list(stimuli.keys())}"
        raise RuntimeError(msg)

    stim_duration = stimuli[stim_key]['flash_interval_sec']

    # from discussion in
    # https://github.com/AllenInstitute/AllenSDK/issues/1572
    #
    # 'flash_interval' contains (stimulus_duration, gray_screen_duration)
    # (as @matchings said above). That second value is redundant with
    # 'blank_duration_range'. I'm not sure what would happen if they were
    # set to be conflicting values in the params. But it looks like
    # they're always consistent. It should always be (0.25, 0.5),
    # except for TRAINING_0 and TRAINING_1, which have statically
    # displayed stimuli (no flashes).

    if stim_duration is None:
        stim_duration = np.NaN
    else:
        stim_duration = stim_duration[0]

    task_parameters['stimulus_duration_sec'] = stim_duration

    task_parameters['omitted_flash_fraction'] = \
        behavior['params'].get('flash_omit_probability', float('nan'))
    task_parameters['response_window_sec'] = \
        [float(x) for x in doc["response_window"]]
    task_parameters['reward_volume'] = config["reward"]["reward_volume"]
    task_parameters['auto_reward_volume'] = doc['auto_reward_volume']
    task_parameters['session_type'] = behavior["params"]["stage"]
    task_parameters['stimulus'] = next(iter(behavior["stimuli"]))
    task_parameters['stimulus_distribution'] = doc["change_time_dist"]

    task_id = config['behavior']['task_id']
    if 'DoC' in task_id:
        task_parameters['task'] = 'change detection'
    else:
        msg = "metadata.get_task_parameters does not "
        msg += f"know how to parse 'task_id' = {task_id}"
        raise RuntimeError(msg)

    n_stimulus_frames = 0
    for stim_type, stim_table in behavior["stimuli"].items():
        n_stimulus_frames += sum(stim_table.get("draw_log", []))
    task_parameters['n_stimulus_frames'] = n_stimulus_frames

    return task_parameters


class BehaviorMetadata(DataObject, InternalMixedReadableMixin,
                       NwbWritableMixin):
    """Container class for behavior metadata"""
    def __init__(self,
                 behavior_session_id: BehaviorSessionId,
                 equipment_name: EquipmentName,
                 sex: Sex,
                 age: Age,
                 stimulus_frame_rate: StimulusFrameRate,
                 session_type: SessionType,
                 date_of_acquisition: DateOfAcquisition,
                 reporter_line: ReporterLine,
                 full_genotype: FullGenotype,
                 behavior_session_uuid: BehaviorSessionUUID,
                 driver_line: DriverLine,
                 mouse_id: MouseId):
        super().__init__(name='behavior_metadata', value=self)
        self._behavior_session_id = behavior_session_id
        self._equipment_name = equipment_name
        self._sex = sex
        self._age = age
        self._stimulus_frame_rate = stimulus_frame_rate
        self._session_type = session_type
        self._date_of_acquisition = date_of_acquisition
        self._reporter_line = reporter_line
        self._full_genotype = full_genotype
        self._behavior_session_uuid = behavior_session_uuid
        self._driver_line = driver_line
        self._mouse_id = mouse_id

        self._exclude_from_equals = set()

    @classmethod
    def from_internal_mixed(
            cls,
            behavior_session_id: BehaviorSessionId,
            stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            lims_db: PostgresQueryMixin
        ) -> "BehaviorMetadata":
        equipment_name = EquipmentName.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        sex = Sex.from_lims(behavior_session_id=behavior_session_id.value,
                            lims_db=lims_db)
        age = Age.from_lims(behavior_session_id=behavior_session_id.value,
                            lims_db=lims_db)
        stimulus_frame_rate = StimulusFrameRate.from_stimulus_file(
            stimulus_timestamps=stimulus_timestamps)
        session_type = SessionType.from_stimulus_file(
            stimulus_file=stimulus_file)
        date_of_acquisition = DateOfAcquisition.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)\
            .validate(stimulus_file=stimulus_file,
                      behavior_session_id=behavior_session_id.value)
        reporter_line = ReporterLine.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        full_genotype = FullGenotype.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)

        foraging_id = ForagingId.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        behavior_session_uuid = BehaviorSessionUUID.from_stimulus_file(
            stimulus_file=stimulus_file)\
            .validate(behavior_session_id=behavior_session_id.value,
                                       foraging_id=foraging_id.value,
                                       stimulus_file=stimulus_file)
        driver_line = DriverLine.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        mouse_id = MouseId.from_lims(
            behavior_session_id=behavior_session_id.value,
                                     lims_db=lims_db)

        return cls(
            behavior_session_id=behavior_session_id,
            equipment_name=equipment_name,
            sex=sex,
            age=age,
            stimulus_frame_rate=stimulus_frame_rate,
            session_type=session_type,
            date_of_acquisition=date_of_acquisition,
            reporter_line=reporter_line,
            full_genotype=full_genotype,
            behavior_session_uuid=behavior_session_uuid,
            driver_line=driver_line,
            mouse_id=mouse_id
        )

    @classmethod
    def from_json(cls, dict_repr: dict) -> "BehaviorMetadata":
        pass

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "BehaviorMetadata":
        pass

    @property
    def equipment_name(self) -> str:
        return self._equipment_name.value

    @property
    def sex(self) -> str:
        return self._sex.value

    @property
    def age_in_days(self) -> Optional[int]:
        return self._age.value

    @property
    def stimulus_frame_rate(self) -> float:
        return self._stimulus_frame_rate.value

    @property
    def session_type(self) -> str:
        return self._session_type.value

    @property
    def date_of_acquisition(self) -> datetime:
        return self._date_of_acquisition.value

    @property
    def reporter_line(self) -> Optional[str]:
        return self._reporter_line.value

    @property
    def indicator(self) -> Optional[str]:
        return self._reporter_line.indicator

    @property
    def full_genotype(self) -> str:
        return self._full_genotype.value

    @property
    def cre_line(self) -> Optional[str]:
        return self._full_genotype.parse_cre_line(warn=True)

    @property
    def behavior_session_uuid(self) -> Optional[uuid.UUID]:
        return self._behavior_session_uuid.value

    @property
    def driver_line(self) -> List[str]:
        return self._driver_line.value

    @property
    def mouse_id(self) -> int:
        return self._mouse_id.value

    @property
    def behavior_session_id(self) -> int:
        return self._behavior_session_id.value

    def to_dict(self) -> dict:
        """Returns dict representation of all properties in class"""
        vars_ = vars(BehaviorMetadata)
        return self._get_properties(vars_=vars_)

    def to_json(self) -> dict:
        pass

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        BehaviorSubject = load_pynwb_extension(SubjectMetadataSchema,
                                               'ndx-aibs-behavior-ophys')
        nwb_subject = BehaviorSubject(
            description="A visual behavior subject with a LabTracks ID",
            age=Age.to_iso8601(age=self.age_in_days),
            driver_line=self.driver_line,
            genotype=self.full_genotype,
            subject_id=str(self.mouse_id),
            reporter_line=self.reporter_line,
            sex=self.sex,
            species='Mus musculus')
        nwbfile.subject = nwb_subject

        nwb_metadata = self._to_nwb()
        extension = self._get_nwb_extension()
        nwb_metadata = extension(**nwb_metadata)
        nwbfile.add_lab_meta_data(nwb_metadata)

        return nwbfile

    @abc.abstractmethod
    def _to_nwb(self) -> dict:
        """Constructs data structure for non-subject metadata"""
        return dict(
            name='metadata',
            behavior_session_id=self.behavior_session_id,
            behavior_session_uuid=str(self.behavior_session_uuid),
            stimulus_frame_rate=self.stimulus_frame_rate,
            session_type=self.session_type,
            equipment_name=self.equipment_name
        )

    @staticmethod
    def _get_nwb_extension():
        return load_pynwb_extension(BehaviorMetadataSchema,
                                                'ndx-aibs-behavior-ophys')

    def _get_properties(self, vars_: dict):
        """Returns all property names and values"""
        return {name: getattr(self, name) for name, value in vars_.items()
                if isinstance(value, property)}

    def __eq__(self, other):
        if not isinstance(other, (BehaviorMetadata, dict)):
            msg = f'Do not know how to compare with type {type(other)}'
            raise NotImplementedError(msg)

        properties_self = self.to_dict()

        if isinstance(other, dict):
            properties_other = other
        else:
            properties_other = other.to_dict()

        for p in properties_self:
            if p in self._exclude_from_equals:
                continue

            x1 = properties_self[p]
            x2 = properties_other[p]

            try:
                compare_session_fields(x1=x1, x2=x2)
            except AssertionError:
                return False
        return True
