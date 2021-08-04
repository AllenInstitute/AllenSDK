import abc
import uuid
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import re
import numpy as np
import pytz

from allensdk.brain_observatory.behavior.session_apis.abcs.\
    data_extractor_base.behavior_data_extractor_base import \
    BehaviorDataExtractorBase
from allensdk.brain_observatory.session_api_utils import compare_session_fields

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


class BehaviorMetadata:
    """Container class for behavior metadata"""
    def __init__(self, extractor: BehaviorDataExtractorBase,
                 stimulus_timestamps: np.ndarray,
                 behavior_stimulus_file: dict):

        self._extractor = extractor
        self._stimulus_timestamps = stimulus_timestamps
        self._behavior_stimulus_file = behavior_stimulus_file
        self._exclude_from_equals = set()

    @property
    def equipment_name(self) -> str:
        return self._extractor.get_equipment_name()

    @property
    def sex(self) -> str:
        return self._extractor.get_sex()

    @property
    def age_in_days(self) -> Optional[int]:
        """Converts the age cod into a numeric days representation"""

        age = self._extractor.get_age()
        return self.parse_age_in_days(age=age, warn=True)

    @property
    def stimulus_frame_rate(self) -> float:
        return self._get_frame_rate(timestamps=self._stimulus_timestamps)

    @property
    def session_type(self) -> str:
        return self._extractor.get_stimulus_name()

    @property
    def date_of_acquisition(self) -> datetime:
        """Return the timestamp for when experiment was started in UTC

        NOTE: This method will only get acquisition datetime from
        extractor (data from LIMS) methods. As a sanity check,
        it will also read the acquisition datetime from the behavior stimulus
        (*.pkl) file and raise a warning if the date differs too much from the
        datetime obtained from the behavior stimulus (*.pkl) file.

        :rtype: datetime
        """
        extractor_acq_date = self._extractor.get_date_of_acquisition()

        pkl_data = self._behavior_stimulus_file
        pkl_raw_acq_date = pkl_data["start_time"]
        if isinstance(pkl_raw_acq_date, datetime):
            pkl_acq_date = pytz.utc.localize(pkl_raw_acq_date)

        elif isinstance(pkl_raw_acq_date, (int, float)):
            # We are dealing with an older pkl file where the acq time is
            # stored as a Unix style timestamp string
            parsed_pkl_acq_date = datetime.fromtimestamp(pkl_raw_acq_date)
            pkl_acq_date = pytz.utc.localize(parsed_pkl_acq_date)
        else:
            pkl_acq_date = None
            warnings.warn(
                "Could not parse the acquisition datetime "
                f"({pkl_raw_acq_date}) found in the following stimulus *.pkl: "
                f"{self._extractor.get_behavior_stimulus_file()}"
            )

        if pkl_acq_date:
            acq_start_diff = (
                    extractor_acq_date - pkl_acq_date).total_seconds()
            # If acquisition dates differ by more than an hour
            if abs(acq_start_diff) > 3600:
                session_id = self._extractor.get_behavior_session_id()
                warnings.warn(
                    "The `date_of_acquisition` field in LIMS "
                    f"({extractor_acq_date}) for behavior session "
                    f"({session_id}) deviates by more "
                    f"than an hour from the `start_time` ({pkl_acq_date}) "
                    "specified in the associated stimulus *.pkl file: "
                    f"{self._extractor.get_behavior_stimulus_file()}"
                )
        return extractor_acq_date

    @property
    def reporter_line(self) -> Optional[str]:
        reporter_line = self._extractor.get_reporter_line()
        return self.parse_reporter_line(reporter_line=reporter_line, warn=True)

    @property
    def cre_line(self) -> Optional[str]:
        """Parses cre_line from full_genotype"""
        cre_line = self.parse_cre_line(full_genotype=self.full_genotype,
                                       warn=True)
        return cre_line

    @property
    def behavior_session_uuid(self) -> Optional[uuid.UUID]:
        """Get the universally unique identifier (UUID)
        """
        data = self._behavior_stimulus_file
        behavior_pkl_uuid = data.get("session_uuid")

        behavior_session_id = self._extractor.get_behavior_session_id()
        foraging_id = self._extractor.get_foraging_id()

        # Sanity check to ensure that pkl file data matches up with
        # the behavior session that the pkl file has been associated with.
        assert_err_msg = (
            f"The behavior session UUID ({behavior_pkl_uuid}) in the "
            f"behavior stimulus *.pkl file "
            f"({self._extractor.get_behavior_stimulus_file()}) does "
            f"does not match the foraging UUID ({foraging_id}) for "
            f"behavior session: {behavior_session_id}")
        assert behavior_pkl_uuid == foraging_id, assert_err_msg

        if behavior_pkl_uuid is None:
            bs_uuid = None
        else:
            bs_uuid = uuid.UUID(behavior_pkl_uuid)
        return bs_uuid

    @property
    def driver_line(self) -> List[str]:
        return sorted(self._extractor.get_driver_line())

    @property
    def mouse_id(self) -> int:
        return self._extractor.get_mouse_id()

    @property
    def full_genotype(self) -> str:
        return self._extractor.get_full_genotype()

    @property
    def behavior_session_id(self) -> int:
        return self._extractor.get_behavior_session_id()

    def get_extractor(self):
        return self._extractor

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Returns dict representation of all properties in class"""
        vars_ = vars(BehaviorMetadata)
        return self._get_properties(vars_=vars_)

    @staticmethod
    def _get_frame_rate(timestamps: np.ndarray):
        return np.round(1 / np.mean(np.diff(timestamps)), 0)

    @staticmethod
    def parse_cre_line(full_genotype: str, warn=False) -> Optional[str]:
        """
        Parameters
        ----------
        full_genotype
            formatted from LIMS, e.g.
            Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt
        warn
            Whether to output warning if parsing fails

        Returns
        ----------
        cre_line
            just the Cre line, e.g. Vip-IRES-Cre, or None if not possible to
            parse
        """
        if ';' not in full_genotype:
            if warn:
                warnings.warn('Unable to parse cre_line from full_genotype')
            return None
        return full_genotype.split(';')[0].replace('/wt', '')

    @staticmethod
    def parse_age_in_days(age: str, warn=False) -> Optional[int]:
        """Converts the age code into a numeric days representation

        Parameters
        ----------
        age
            age code, ie P123
        warn
            Whether to output warning if parsing fails
        """
        if not age.startswith('P'):
            if warn:
                warnings.warn('Could not parse numeric age from age code '
                              '(age code does not start with "P")')
            return None

        match = re.search(r'\d+', age)

        if match is None:
            if warn:
                warnings.warn('Could not parse numeric age from age code '
                              '(no numeric values found in age code)')
            return None

        start, end = match.span()
        return int(age[start:end])

    @staticmethod
    def parse_reporter_line(reporter_line: Optional[List[str]],
                            warn=False) -> Optional[str]:
        """There can be multiple reporter lines, so it is returned from LIMS
        as a list. But there shouldn't be more than 1 for behavior. This
        tries to convert to str

        Parameters
        ----------
        reporter_line
            List of reporter line
        warn
            Whether to output warnings if parsing fails

        Returns
        ---------
        single reporter line, or None if not possible
        """
        if reporter_line is None:
            if warn:
                warnings.warn('Error parsing reporter line. It is null.')
            return None

        if len(reporter_line) == 0:
            if warn:
                warnings.warn('Error parsing reporter line. '
                              'The array is empty')
            return None

        if isinstance(reporter_line, str):
            return reporter_line

        if len(reporter_line) > 1:
            if warn:
                warnings.warn('More than 1 reporter line. Returning the first '
                              'one')

        return reporter_line[0]

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

    @staticmethod
    def parse_indicator(reporter_line: Optional[str], warn=False) -> Optional[
            str]:
        """Parses indicator from reporter"""
        reporter_substring_indicator_map = {
            'GCaMP6f': 'GCaMP6f',
            'GC6f': 'GCaMP6f',
            'GCaMP6s': 'GCaMP6s'
        }
        if reporter_line is None:
            if warn:
                warnings.warn(
                    'Could not parse indicator from reporter because '
                    'there is no reporter')
            return None

        for substr, indicator in reporter_substring_indicator_map.items():
            if substr in reporter_line:
                return indicator

        if warn:
            warnings.warn(
                'Could not parse indicator from reporter because none'
                'of the expected substrings were found in the reporter')
        return None
