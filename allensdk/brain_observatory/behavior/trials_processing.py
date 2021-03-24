from typing import List, Dict
import uuid
from copy import deepcopy
import collections
import dateutil

import pandas as pd
import numpy as np

from allensdk import one
from allensdk.brain_observatory.behavior.dprime import (
    get_rolling_dprime, get_trial_count_corrected_false_alarm_rate,
    get_trial_count_corrected_hit_rate,
    get_hit_rate, get_false_alarm_rate)

# TODO: add trial column descriptions
TRIAL_COLUMN_DESCRIPTION_DICT = {}

EDF_COLUMNS = ['index', 'lick_times', 'auto_rewarded', 'cumulative_volume',
               'cumulative_reward_number', 'reward_volume', 'reward_times',
               'reward_frames', 'rewarded', 'optogenetics', 'response_type',
               'response_time', 'change_time', 'change_frame',
               'response_latency', 'starttime', 'startframe', 'trial_length',
               'scheduled_change_time', 'endtime', 'endframe',
               'initial_image_category', 'initial_image_name',
               'change_image_name', 'change_image_category', 'change_ori',
               'change_contrast', 'initial_ori', 'initial_contrast',
               'delta_ori', 'mouse_id', 'response_window', 'task', 'stage',
               'session_duration', 'user_id', 'LDT_mode',
               'blank_screen_timeout', 'stim_duration',
               'blank_duration_range', 'prechange_minimum',
               'stimulus_distribution', 'stimulus', 'distribution_mean',
               'computer_name', 'behavior_session_uuid', 'startdatetime',
               'date', 'year', 'month', 'day', 'hour', 'dayofweek',
               'number_of_rewards', 'rig_id', 'trial_type',
               'lick_frames', 'reward_licks', 'reward_lick_count',
               'reward_lick_latency', 'reward_rate', 'response', 'color']

RIG_NAME = {
    'W7DTMJ19R2F': 'A1',
    'W7DTMJ35Y0T': 'A2',
    'W7DTMJ03J70R': 'Dome',
    'W7VS-SYSLOGIC2': 'A3',
    'W7VS-SYSLOGIC3': 'A4',
    'W7VS-SYSLOGIC4': 'A5',
    'W7VS-SYSLOGIC5': 'A6',
    'W7VS-SYSLOGIC7': 'B1',
    'W7VS-SYSLOGIC8': 'B2',
    'W7VS-SYSLOGIC9': 'B3',
    'W7VS-SYSLOGIC10': 'B4',
    'W7VS-SYSLOGIC11': 'B5',
    'W7VS-SYSLOGIC12': 'B6',
    'W7VS-SYSLOGIC13': 'C1',
    'W7VS-SYSLOGIC14': 'C2',
    'W7VS-SYSLOGIC15': 'C3',
    'W7VS-SYSLOGIC16': 'C4',
    'W7VS-SYSLOGIC17': 'C5',
    'W7VS-SYSLOGIC18': 'C6',
    'W7VS-SYSLOGIC19': 'D1',
    'W7VS-SYSLOGIC20': 'D2',
    'W7VS-SYSLOGIC21': 'D3',
    'W7VS-SYSLOGIC22': 'D4',
    'W7VS-SYSLOGIC23': 'D5',
    'W7VS-SYSLOGIC24': 'D6',
    'W7VS-SYSLOGIC31': 'E1',
    'W7VS-SYSLOGIC32': 'E2',
    'W7VS-SYSLOGIC33': 'E3',
    'W7VS-SYSLOGIC34': 'E4',
    'W7VS-SYSLOGIC35': 'E5',
    'W7VS-SYSLOGIC36': 'E6',
    'W7DT102905': 'F1',
    'W10DT102905': 'F1',
    'W7DT102904': 'F2',
    'W7DT102903': 'F3',
    'W7DT102914': 'F4',
    'W7DT102913': 'F5',
    'W7DT12497': 'F6',
    'W7DT102906': 'G1',
    'W7DT102907': 'G2',
    'W7DT102908': 'G3',
    'W7DT102909': 'G4',
    'W7DT102910': 'G5',
    'W7DT102911': 'G6',
    'W7VS-SYSLOGIC26': 'Widefield-329',
    'OSXLTTF6T6.local': 'DougLaptop',
    'W7DTMJ026LUL': 'DougPC',
    'W7DTMJ036PSL': 'Marina2P_Sutter',
    'W7DT2PNC1STIM': '2P6',
    'W7DTMJ234MG': 'peterl_2p',
    'W7DT2P3STiM': '2P3',
    'W7DT2P4STIM': '2P4',
    'W7DT2P5STIM': '2P5',
    'W10DTSM118296': 'NP3',
    'meso1stim': 'MS1',
    'localhost': 'localhost'
}
RIG_NAME = {k.lower(): v for k, v in RIG_NAME.items()}
COMPUTER_NAME = dict((v, k) for k, v in RIG_NAME.items())


def resolve_initial_image(stimuli, start_frame):
    """Attempts to resolve the initial image for a given start_frame for a trial

    Parameters
    ----------
    stimuli: Mapping
        foraging2 shape stimuli mapping
    start_frame: int
        start frame of the trial

    Returns
    -------
    initial_image_category_name: str
        stimulus category of initial image
    initial_image_group: str
        group name of the initial image
    initial_image_name: str
        name of the initial image
    """
    max_frame = float("-inf")
    initial_image_group = ''
    initial_image_name = ''
    initial_image_category_name = ''

    for stim_category_name, stim_dict in stimuli.items():
        for set_event in stim_dict["set_log"]:
            set_frame = set_event[3]
            if set_frame <= start_frame and set_frame >= max_frame:
                # hack assumes initial_image_group == initial_image_name,
                # only initial_image_name is present for natual_scenes
                initial_image_group = initial_image_name = set_event[1]
                initial_image_category_name = stim_category_name
                if initial_image_category_name == 'grating':
                    initial_image_name = f'gratings_{initial_image_name}'
                max_frame = set_frame

    return initial_image_category_name, initial_image_group, initial_image_name


def trial_data_from_log(trial):
    '''
    Infer trial logic from trial log. Returns a dictionary.

    * reward volume: volume of water delivered on the trial, in mL

    Each of the following values is boolean:

    Trial category values are mutually exclusive
    * go: trial was a go trial (trial with a stimulus change)
    * catch: trial was a catch trial (trial with a sham stimulus change)

    stimulus_change/sham_change are mutually exclusive
    * stimulus_change: did the stimulus change (True on 'go' trials)
    * sham_change: stimulus did not change, but response was evaluated
                   (True on 'catch' trials)

    Each trial can be one (and only one) of the following:
    * hit (stimulus changed, animal responded in response window)
    * miss (stimulus changed, animal did not respond in response window)
    * false_alarm (stimulus did not change,
                   animal responded in response window)
    * correct_reject (stimulus did not change,
                      animal did not respond in response window)
    * aborted (animal responded before change time)
    * auto_rewarded (reward was automatically delivered following the change.
                     This will bias the animals choice and should not be
                     categorized as hit/miss)
    '''
    trial_event_names = [val[0] for val in trial['events']]
    hit = 'hit' in trial_event_names
    false_alarm = 'false_alarm' in trial_event_names
    miss = 'miss' in trial_event_names
    sham_change = 'sham_change' in trial_event_names
    stimulus_change = 'stimulus_changed' in trial_event_names
    aborted = 'abort' in trial_event_names

    if aborted:
        go = catch = auto_rewarded = False
    else:
        catch = trial["trial_params"]["catch"] is True
        auto_rewarded = trial["trial_params"]["auto_reward"]
        go = not catch and not auto_rewarded

    correct_reject = catch and not false_alarm

    if auto_rewarded:
        hit = miss = correct_reject = false_alarm = False

    return {
        "reward_volume": sum([r[0] for r in trial.get("rewards", [])]),
        "hit": hit,
        "false_alarm": false_alarm,
        "miss": miss,
        "sham_change": sham_change,
        "stimulus_change": stimulus_change,
        "aborted": aborted,
        "go": go,
        "catch": catch,
        "auto_rewarded": auto_rewarded,
        "correct_reject": correct_reject,
    }


def validate_trial_condition_exclusivity(trial_index, **trial_conditions):
    '''ensure that only one of N possible mutually
    exclusive trial conditions is True'''
    on = []
    for condition, value in trial_conditions.items():
        if value:
            on.append(condition)

    if len(on) != 1:
        all_conditions = list(trial_conditions.keys())
        msg = f"expected exactly 1 trial condition out of {all_conditions} "
        msg += f"to be True, instead {on} were True (trial {trial_index})"
        raise AssertionError(msg)


def get_trial_reward_time(rebased_reward_times,
                          start_time,
                          stop_time):
    '''extract reward times in time range'''
    reward_times = rebased_reward_times[np.where(np.logical_and(
        rebased_reward_times >= start_time,
        rebased_reward_times <= stop_time
    ))]
    return float('nan') if len(reward_times) == 0 else one(reward_times)


def _get_response_time(licks: List[float], aborted: bool) -> float:
    """
    Return the time the first lick occurred in a non-"aborted" trial.
    A response time is not returned for on an "aborted trial", since by
    definition, the animal licked before the change stimulus.

    Parameters
    ==========
    licks: List[float]
        List of timestamps that a lick occurred during this trial.
        The list should contain all licks that occurred while the trial
        was active (between 'trial_start' and 'trial_end' events)
    aborted: bool
        Whether or not the trial was "aborted". This means that the
        response occurred before the stimulus change and should not be
        a valid response.
    Returns
    =======
    float
        Time of first lick if there was a valid response, otherwise
        NaN. See rules above.
    """
    if aborted:
        return float("nan")
    if len(licks):
        return licks[0]
    else:
        return float("nan")


def get_trial_timing(
        event_dict: dict,
        licks: List[float], go: bool, catch: bool, auto_rewarded: bool,
        hit: bool, false_alarm: bool, aborted: bool,
        timestamps: np.ndarray,
        monitor_delay: float):
    """
    Extract a dictionary of trial timing data.
    See trial_data_from_log for a description of the trial types.

    Parameters
    ==========
    event_dict: dict
        Dictionary of trial events in the well-known `pkl` file
    licks: List[float]
        list of lick timestamps, from the `get_licks` response for
        the BehaviorOphysExperiment.api.
    go: bool
        True if "go" trial, False otherwise. Mutually exclusive with
        `catch`.
    catch: bool
        True if "catch" trial, False otherwise. Mutually exclusive
        with `go.`
    auto_rewarded: bool
        True if "auto_rewarded" trial, False otherwise.
    hit: bool
        True if "hit" trial, False otherwise
    false_alarm: bool
        True if "false_alarm" trial, False otherwise
    aborted: bool
        True if "aborted" trial, False otherwise
    timestamps: np.ndarray[1d]
        Array of ground truth timestamps for the session
        (sync times, if available)
    monitor_delay: float
        The monitor delay in seconds associated with the session

    Returns
    =======
    dict
        start_time: float
            The time the trial started (in seconds elapsed from
            recording start)
        stop_time: float
            The time the trial ended (in seconds elapsed from
            recording start)
        trial_length: float
            Duration of the trial in seconds
        response_time: float
            The response time, for non-aborted trials. This is equal
            to the first lick in the trial. For aborted trials or trials
            without licks, `response_time` is NaN.
        change_frame: int
            The frame number that the stimulus changed
        change_time: float
            The time in seconds that the stimulus changed
        response_latency: float or None
            The time in seconds between the stimulus change and the
            animal's lick response, if the trial is a "go", "catch", or
            "auto_rewarded" type. If the animal did not respond,
            return `float("inf")`. In all other cases, return None.

    Notes
    =====
    The following parameters are mutually exclusive (exactly one can
    be true):
        hit, miss, false_alarm, aborted, auto_rewarded
    """
    assert not (aborted and (hit or false_alarm or auto_rewarded)), (
        "'aborted' trials cannot be 'hit', 'false_alarm', or 'auto_rewarded'")
    assert not (hit and false_alarm), (
        "both `hit` and `false_alarm` cannot be True, they are mutually "
        "exclusive categories")
    assert not (go and catch), (
        "both `go` and `catch` cannot be True, they are mutually exclusive "
        "categories")
    assert not (go and auto_rewarded), (
        "both `go` and `auto_rewarded` cannot be True, they are mutually "
        "exclusive categories")

    start_time = event_dict["trial_start", ""]['timestamp']
    stop_time = event_dict["trial_end", ""]['timestamp']

    response_time = _get_response_time(licks, aborted)

    if go or auto_rewarded:
        change_frame = event_dict.get(('stimulus_changed', ''))['frame']
        change_time = timestamps[change_frame] + monitor_delay
    elif catch:
        change_frame = event_dict.get(('sham_change', ''))['frame']
        change_time = timestamps[change_frame] + monitor_delay
    else:
        change_time = float("nan")
        change_frame = float("nan")

    if not (go or catch or auto_rewarded):
        response_latency = None
    elif len(licks) > 0:
        response_latency = licks[0] - change_time
    else:
        response_latency = float("inf")

    return {
        "start_time": start_time,
        "stop_time": stop_time,
        "trial_length": stop_time - start_time,
        "response_time": response_time,
        "change_frame": change_frame,
        "change_time": change_time,
        "response_latency": response_latency,
    }


def get_trial_image_names(trial, stimuli) -> Dict[str, str]:
    """
    Gets the name of the stimulus presented at the beginning of the trial and
    what is it changed to at the end of the trial.
    Parameters
    ----------
    trial: A trial in a behavior ophys session
    stimuli: The stimuli presentation log for the behavior session

    Returns
    -------
        A dictionary indicating the starting_stimulus and what the stimulus is
        changed to.

    """
    grating_oris = {'horizontal', 'vertical'}
    trial_start_frame = trial["events"][0][3]
    initial_image_category_name, _, initial_image_name = resolve_initial_image(
        stimuli, trial_start_frame)
    if len(trial["stimulus_changes"]) == 0:
        change_image_name = initial_image_name
    else:
        ((from_set, from_name),
         (to_set, to_name),
         _, _) = trial["stimulus_changes"][0]

        # do this to fix names if the stimuli is a grating
        if from_set in grating_oris:
            from_name = f'gratings_{from_name}'
        if to_set in grating_oris:
            to_name = f'gratings_{to_name}'
        assert from_name == initial_image_name
        change_image_name = to_name

    return {
        "initial_image_name": initial_image_name,
        "change_image_name": change_image_name
    }


def get_trial_bounds(trial_log: List) -> List:
    """
    Adjust trial boundaries from a trial_log so that there is no dead time
    between trials.

    Parameters
    ----------
    trial_log: list
        The trial_log read in from the well known behavior stimulus pickle file

    Returns
    -------
    list
        Each element in the list is a tuple of the form
        (start_frame, end_frame) so that the ith element
        of the list gives the start and end frames of
        the ith trial. The endframe of the last trial will
        be -1, indicating that it should map to the last
        timestamp in the session
    """
    start_frames = []

    for trial in trial_log:
        start_f = None
        for event in trial['events']:
            if event[0] == 'trial_start':
                start_f = event[-1]
                break
        if start_f is None:
            msg = "Could not find a 'trial_start' event "
            msg += "for all trials in the trial log\n"
            msg += f"{trial}"
            raise ValueError(msg)

        if len(start_frames) > 0 and start_f < start_frames[-1]:
            msg = "'trial_start' frames in trial log "
            msg += "are not in ascending order"
            msg += f"\ntrial_log: {trial_log}"
            raise ValueError(msg)

        start_frames.append(start_f)

    end_frames = [idx for idx in start_frames[1:]+[-1]]
    return list([(s, e) for s, e in zip(start_frames, end_frames)])


def get_trials_from_data_transform(input_transform) -> pd.DataFrame:
    """
    Create and return a pandas DataFrame containing data about
    the trials associated with this session

    Parameters
    ----------
    input_transform:
        An instantiation of a class that inherits from either
        BehaviorDataTransform or BehaviorOphysDataTransform.
        This object will be used to get at the data needed by
        this method to create the trials dataframe.

    Returns
    -------
    pd.DataFrame
           A dataframe containing data pertaining to the trials that
           make up this session

    Notes
    -----
    The input_transform object must have the following methods:

    input_transform._behavior_stimulus_file
        Which returns the dict resulting from reading in this session's
        stimulus_data pickle file

    input_transform.get_rewards
        Which returns a dataframe containing data about rewards given
        during this session, i.e. the output of
        allensdk/brain_observatory/behavior/rewards_processing.get_rewards

    input_transform.get_licks
        Which returns a dataframe containing the columns `time` and `frame`
        denoting the time (in seconds) and frame number at which licks
        occurred during this session

    input_transform.get_stimulus_timestamps
        Which returns a numpy.ndarray of timestamps (in seconds) associated
        with the frames presented in this session.

    input_transform.get_monitor_delay
        Which returns the monitory delay (in seconds) associated with the
        experimental rig
    """

    missing_data_streams = []
    for method_name in ('get_rewards', 'get_licks',
                        'get_stimulus_timestamps',
                        'get_monitor_delay',
                        '_behavior_stimulus_file'):
        if not hasattr(input_transform, method_name):
            missing_data_streams.append(method_name)
    if len(missing_data_streams) > 0:
        msg = 'Cannot run trials_processing.get_trials\n'
        msg += 'The object you passed as input is missing '
        msg += 'the following required methods:\n'
        for method_name in missing_data_streams:
            msg += f'{method_name}\n'
        raise ValueError(msg)

    rewards_df = input_transform.get_rewards()
    licks_df = input_transform.get_licks()
    timestamps = input_transform.get_stimulus_timestamps()
    monitor_delay = input_transform.get_monitor_delay()
    data = input_transform._behavior_stimulus_file()

    stimuli = data["items"]["behavior"]["stimuli"]
    trial_log = data["items"]["behavior"]["trial_log"]

    trial_bounds = get_trial_bounds(trial_log)

    all_trial_data = [None] * len(trial_log)
    lick_frames = licks_df.frame.values
    reward_times = rewards_df['timestamps'].values

    for idx, trial in enumerate(trial_log):
        # match each event in the trial log to the sync timestamps
        event_dict = {(e[0], e[1]): {'timestamp': timestamps[e[3]],
                                     'frame': e[3]}
                      for e in trial['events']}

        tr_data = {"trial": trial["index"]}

        trial_start = trial_bounds[idx][0]
        trial_end = trial_bounds[idx][1]

        # this block of code is trying to mimic
        # https://github.com/AllenInstitute/visual_behavior_analysis/blob/master/visual_behavior/translator/foraging2/__init__.py#L377-L381
        # https://github.com/AllenInstitute/visual_behavior_analysis/blob/master/visual_behavior/translator/foraging2/extract_movies.py#L59-L94
        # https://github.com/AllenInstitute/visual_behavior_analysis/blob/master/visual_behavior/translator/core/annotate.py#L11-L36
        #
        # In summary: there are cases where an "epilogue movie" is shown
        # after the proper stimuli; we do not want licks that occur
        # during this epilogue movie to be counted as belonging to
        # the last trial
        # https://github.com/AllenInstitute/visual_behavior_analysis/issues/482

        if trial_end < 0:
            bhv = data['items']['behavior']['items']
            if 'fingerprint' in bhv.keys():
                trial_end = bhv['fingerprint']['starting_frame']

        # select licks that fall between trial_start and trial_end;
        # licks on the boundary get assigned to the trial that is ending,
        # rather than the trial that is starting
        if trial_end > 0:
            valid_idx = np.where(np.logical_and(lick_frames > trial_start,
                                                lick_frames <= trial_end))
        else:
            valid_idx = np.where(lick_frames > trial_start)

        valid_licks = lick_frames[valid_idx]
        if len(valid_licks) > 0:
            tr_data["lick_times"] = timestamps[valid_licks]
        else:
            tr_data["lick_times"] = np.array([], dtype=float)

        tr_data["reward_time"] = get_trial_reward_time(
            reward_times,
            event_dict[('trial_start', '')]['timestamp'],
            event_dict[('trial_end', '')]['timestamp']
        )
        tr_data.update(trial_data_from_log(trial))
        tr_data.update(get_trial_timing(
            event_dict,
            tr_data['lick_times'],
            tr_data['go'],
            tr_data['catch'],
            tr_data['auto_rewarded'],
            tr_data['hit'],
            tr_data['false_alarm'],
            tr_data["aborted"],
            timestamps,
            monitor_delay
        ))
        tr_data.update(get_trial_image_names(trial, stimuli))

        # ensure that only one trial condition is True
        # (they are mutually exclusive)
        condition_dict = {}
        for key in ['hit',
                    'miss',
                    'false_alarm',
                    'correct_reject',
                    'auto_rewarded',
                    'aborted']:
            condition_dict[key] = tr_data[key]
        validate_trial_condition_exclusivity(idx, **condition_dict)

        all_trial_data[idx] = tr_data

    trials = pd.DataFrame(all_trial_data).set_index('trial')
    trials.index = trials.index.rename('trials_id')
    del trials["sham_change"]

    return trials


def local_time(iso_timestamp, timezone=None):
    datetime = pd.to_datetime(iso_timestamp)
    if not datetime.tzinfo:
        tzinfo = dateutil.tz.gettz('America/Los_Angeles')
        datetime = datetime.replace(tzinfo=tzinfo)
    return datetime.isoformat()


def get_time(exp_data):
    vsyncs = exp_data["items"]["behavior"]["intervalsms"]
    return np.hstack((0, vsyncs)).cumsum() / 1000.0


def data_to_licks(data, time):
    lick_frames = data['items']['behavior']['lick_sensors'][0]['lick_events']
    lick_times = time[lick_frames]
    return pd.DataFrame(data={"timestamps": lick_times,
                              "frame": lick_frames})


def get_mouse_id(exp_data):
    return exp_data["items"]["behavior"]['config']['behavior']['mouse_id']


def get_params(exp_data):

    params = deepcopy(exp_data["items"]["behavior"].get("params", {}))
    params.update(exp_data["items"]["behavior"].get("cl_params", {}))

    if "response_window" in params:
        # tuple to list
        params["response_window"] = list(params["response_window"])

    return params


def get_even_sampling(data):
    """Get status of even_sampling

    Parameters
    ----------
    data: Mapping
        foraging2 experiment output data

    Returns
    -------
    bool:
        True if even_sampling is enabled
    """

    stimuli = data['items']['behavior']['stimuli']
    for stimuli_group_name, stim in stimuli.items():
        if (stim['obj_type'].lower() == 'docimagestimulus'
           and stim['sampling'] in ['even', 'file']):

            return True

    return False


def data_to_metadata(data, time):

    config = data['items']['behavior']['config']
    doc = config['DoC']
    stimuli = data['items']['behavior']['stimuli']

    metadata = {
        "startdatetime": local_time(data["start_time"],
                                    timezone='America/Los_Angeles'),
        "rig_id": RIG_NAME.get(data['platform_info']['computer_name'].lower(),
                               'unknown'),
        "computer_name": data['platform_info']['computer_name'],
        "reward_vol": config["reward"]["reward_volume"],
        "auto_reward_vol": doc["auto_reward_volume"],
        "params": get_params(data),
        "mouseid": config['behavior']['mouse_id'],
        "response_window": list(data["items"]["behavior"].get("config", {}).get("DoC", {}).get("response_window")),  # noqa: E501
        "task": config["behavior"]['task_id'],
        "stage": data["items"]["behavior"]["params"]["stage"],
        "stoptime": time[-1] - time[0],
        "userid": data["items"]["behavior"]['cl_params']['user_id'],
        "lick_detect_training_mode": "single",
        "blankscreen_on_timeout": False,
        "stim_duration": doc['stimulus_window'] * 1000,
        "blank_duration_range": list(doc['blank_duration_range']),
        "delta_minimum": doc['pre_change_time'],
        "stimulus_distribution": doc["change_time_dist"],
        "delta_mean": doc["change_time_scale"],
        "trial_duration": None,
        "n_stimulus_frames": sum([sum(s.get("draw_log", []))
                                 for s in stimuli.values()]),
        "stimulus": list(stimuli.keys())[0],
        "warm_up_trials": doc["warm_up_trials"],
        "stimulus_window": doc["stimulus_window"],
        "volume_limit": config["behavior"]["volume_limit"],
        "failure_repeats": doc["failure_repeats"],
        "catch_frequency": doc["catch_freq"],
        "auto_reward_delay": doc.get("auto_reward_delay", 0.0),
        "free_reward_trials": doc["free_reward_trials"],
        "min_no_lick_time": doc["min_no_lick_time"],
        "max_session_duration": doc["max_task_duration_min"],
        "abort_on_early_response": doc["abort_on_early_response"],
        "initial_blank_duration": doc["initial_blank"],
        "even_sampling_enabled": get_even_sampling(data),
        "behavior_session_uuid": uuid.UUID(data["session_uuid"]),
        "periodic_flash": doc['periodic_flash'],
        "platform_info": data['platform_info']
    }

    return metadata


def get_response_latency(change_event, trial):

    for response_event in trial['events']:
        if response_event[0] in ['hit', 'false_alarm']:
            return response_event[2] - change_event[2]
    return float('inf')


def get_change_time_frame_response_latency(trial):

    for change_event in trial['events']:
        if change_event[0] in ['stimulus_changed', 'sham_change']:
            return (change_event[2],
                    change_event[3],
                    get_response_latency(change_event, trial))
    return None, None, None


def get_stimulus_attr_changes(stim_dict,
                              change_frame,
                              first_frame,
                              last_frame):
    """
    Notes
    -----
    - assumes only two stimuli are ever shown
    - converts attr_names to lowercase
    - gets the net attr changes from the start of a trial to the end of a trial
    """
    initial_attr = {}
    change_attr = {}

    for attr_name, set_value, set_time, set_frame in stim_dict["set_log"]:
        if set_frame <= first_frame:
            initial_attr[attr_name.lower()] = set_value
        elif set_frame <= last_frame:
            change_attr[attr_name.lower()] = set_value
        else:
            pass

    return initial_attr, change_attr


def get_image_info_from_trial(trial_log, ti):

    if ti == -1:
        raise RuntimeError('Should not have been possible')

    if len(trial_log[ti]["stimulus_changes"]) == 1:

        ((from_group, from_name, ),
         (to_group, to_name),
         _, _) = trial_log[ti]["stimulus_changes"][0]

        return from_group, from_name, to_group, to_name
    else:

        (_, _,
         prev_group,
         prev_name) = get_image_info_from_trial(trial_log, ti - 1)

        return prev_group, prev_name, prev_group, prev_name


def get_ori_info_from_trial(trial_log, ti, ):
    if ti == -1:
        raise IndexError('No change on first trial.')

    if len(trial_log[ti]["stimulus_changes"]) == 1:

        ((initial_group, initial_orientation),
         (change_group, change_orientation, ),
         _, _) = trial_log[ti]["stimulus_changes"][0]

        return change_orientation, change_orientation, None
    else:
        return get_ori_info_from_trial(trial_log, ti - 1)


def get_trials_v0(data, time):
    stimuli = data["items"]["behavior"]["stimuli"]
    if len(list(stimuli.keys())) != 1:
        raise ValueError('Only one stimuli supported.')

    stim_name, stim = next(iter(stimuli.items()))
    if stim_name not in ['images', 'grating', ]:
        raise ValueError('Unsupported stimuli name: {}.'.format(stim_name))

    doc = data["items"]["behavior"]["config"]["DoC"]

    implied_type = stim["obj_type"]
    trial_log = data["items"]["behavior"]["trial_log"]
    pre_change_time = doc['pre_change_time']
    initial_blank_duration = doc["initial_blank"]

    # we need this for the situations where a
    # change doesn't occur in the first trial
    initial_stim = stim['set_log'][0]

    trials = collections.defaultdict(list)
    for ti, trial in enumerate(trial_log):

        trials['index'].append(trial["index"])
        trials['lick_times'].append([lick[0] for lick in trial["licks"]])
        trials['auto_rewarded'].append(trial["trial_params"]["auto_reward"]
                                       if not trial['trial_params']['catch']
                                       else None)

        trials['cumulative_volume'].append(trial["cumulative_volume"])
        trials['cumulative_reward_number'].append(trial["cumulative_rewards"])

        trials['reward_volume'].append(sum([r[0]
                                       for r in trial.get("rewards", [])]))

        trials['reward_times'].append([reward[1]
                                       for reward in trial["rewards"]])

        trials['reward_frames'].append([reward[2]
                                        for reward in trial["rewards"]])

        trials['rewarded'].append(trial["trial_params"]["catch"] is False)
        trials['optogenetics'].append(trial["trial_params"].get("optogenetics", False))  # noqa: E501
        trials['response_type'].append([])
        trials['response_time'].append([])
        trials['change_time'].append(get_change_time_frame_response_latency(trial)[0])  # noqa: E501
        trials['change_frame'].append(get_change_time_frame_response_latency(trial)[1])  # noqa: E501
        trials['response_latency'].append(get_change_time_frame_response_latency(trial)[2])  # noqa: E501
        trials['starttime'].append(trial["events"][0][2])
        trials['startframe'].append(trial["events"][0][3])
        trials['trial_length'].append(trial["events"][-1][2] -
                                      trial["events"][0][2])
        trials['scheduled_change_time'].append(pre_change_time +
                                               initial_blank_duration +
                                               trial["trial_params"]["change_time"])  # noqa: E501
        trials['endtime'].append(trial["events"][-1][2])
        trials['endframe'].append(trial["events"][-1][3])

        # Stimulus:
        if implied_type == 'DoCImageStimulus':
            (from_group,
             from_name,
             to_group,
             to_name) = get_image_info_from_trial(trial_log, ti)
            trials['initial_image_name'].append(from_name)
            trials['initial_image_category'].append(from_group)
            trials['change_image_name'].append(to_name)
            trials['change_image_category'].append(to_group)
            trials['change_ori'].append(None)
            trials['change_contrast'].append(None)
            trials['initial_ori'].append(None)
            trials['initial_contrast'].append(None)
            trials['delta_ori'].append(None)
        elif implied_type == 'DoCGratingStimulus':
            try:
                (change_orientation,
                 initial_orientation,
                 delta_orientation) = get_ori_info_from_trial(trial_log, ti)
            except IndexError:
                # shape: group_name, orientation,
                #        stimulus time relative to start, frame
                orientation = initial_stim[1]
                change_orientation = orientation
                initial_orientation = orientation
                delta_orientation = None
            trials['initial_image_category'].append('')
            trials['initial_image_name'].append('')
            trials['change_image_name'].append('')
            trials['change_image_category'].append('')
            trials['change_ori'].append(change_orientation)
            trials['change_contrast'].append(None)
            trials['initial_ori'].append(initial_orientation)
            trials['initial_contrast'].append(None)
            trials['delta_ori'].append(delta_orientation)
        else:
            msg = 'Unsupported stimulus type: {}'.format(implied_type)
            raise NotImplementedError(msg)

    return pd.DataFrame(trials)


def categorize_one_trial(tr):
    if pd.isnull(tr['change_time']):
        if (len(tr['lick_times']) > 0):
            trial_type = 'aborted'
        else:
            trial_type = 'other'
    else:
        if (tr['auto_rewarded'] is True):
            return 'autorewarded'
        elif (tr['rewarded'] is True):
            return 'go'
        elif (tr['rewarded'] == 0):
            return 'catch'
        else:
            return 'other'
    return trial_type


def find_licks(reward_times, licks, window=3.5):
    if len(reward_times) == 0:
        return []
    else:
        reward_time = one(reward_times)
        reward_lick_mask = ((licks['timestamps'] > reward_time) &
                            (licks['timestamps'] < (reward_time + window)))

        tr_licks = licks[reward_lick_mask].copy()
        tr_licks['timestamps'] -= reward_time
        return tr_licks['timestamps'].values


def calculate_reward_rate(response_latency=None,
                          starttime=None,
                          window=0.75,
                          trial_window=25,
                          initial_trials=10):

    assert len(response_latency) == len(starttime)

    df = pd.DataFrame({'response_latency': response_latency,
                       'starttime': starttime})

    # adds a column called reward_rate to the input dataframe
    # the reward_rate column contains a rolling average of rewards/min
    # window sets the window in which a response is considered correct,
    # so a window of 1.0 means licks before 1.0 second are considered correct
    #
    # Reorganized into this unit-testable form by Nick Cain April 25 2019

    reward_rate = np.zeros(len(df))
    # make the initial reward rate infinite,
    # so that you include the first trials automatically.
    reward_rate[:initial_trials] = np.inf

    for trial_number in range(initial_trials, len(df)):

        min_index = np.max((0, trial_number - trial_window))
        max_index = np.min((trial_number + trial_window, len(df)))
        df_roll = df.iloc[min_index:max_index]

        # get a rolling number of correct trials
        correct = len(df_roll[df_roll.response_latency < window])

        # get the time elapsed over the trials
        time_elapsed = df_roll.starttime.iloc[-1] - df_roll.starttime.iloc[0]

        # calculate the reward rate, rewards/min
        reward_rate_on_this_lap = correct / time_elapsed * 60

        reward_rate[trial_number] = reward_rate_on_this_lap
    return reward_rate


def get_response_type(trials):

    response_type = []
    for idx in trials.index:
        if trials.loc[idx].trial_type.lower() == 'aborted':
            response_type.append('EARLY_RESPONSE')
        elif (trials.loc[idx].rewarded) & (trials.loc[idx].response == 1):
            response_type.append('HIT')
        elif (trials.loc[idx].rewarded) & (trials.loc[idx].response != 1):
            response_type.append('MISS')
        elif (not trials.loc[idx].rewarded) & (trials.loc[idx].response == 1):
            response_type.append('FA')
        elif (not trials.loc[idx].rewarded) & (trials.loc[idx].response != 1):
            response_type.append('CR')
        else:
            response_type.append('other')

    return response_type


def colormap(trial_type, response_type):

    if trial_type == 'aborted':
        return 'lightgray'

    if trial_type == 'autorewarded':
        return 'darkblue'

    if trial_type == 'go':
        if response_type == 'HIT':
            return '#55a868'
        return '#ccb974'

    if trial_type == 'catch':
        if response_type == 'FA':
            return '#c44e52'
        return '#4c72b0'


def create_extended_trials(trials=None, metadata=None, time=None, licks=None):

    startdatetime = dateutil.parser.parse(metadata['startdatetime'])
    edf = trials[~pd.isnull(trials['reward_times'])].reset_index(drop=True).copy()  # noqa: E501

    # Buggy computation of trial_length (for backwards compatibility)
    edf.drop(['trial_length'], axis=1, inplace=True)

    edf['endtime_buggy'] = [edf['starttime'].iloc[ti + 1]
                            if ti < len(edf) - 1
                            else time[-1]
                            for ti in range(len(edf))]

    edf['trial_length'] = edf['endtime_buggy'] - edf['starttime']
    edf.drop(['endtime_buggy'], axis=1, inplace=True)

    # Make trials contiguous, and rebase time:
    edf.drop(['endframe',
              'starttime',
              'endtime',
              'change_time',
              'lick_times',
              'reward_times'], axis=1, inplace=True)

    edf['endframe'] = [edf['startframe'].iloc[ti + 1]
                       if ti < len(edf) - 1
                       else len(time) - 1
                       for ti in range(len(edf))]

    _lks = licks['frame']
    edf['lick_frames'] = [_lks[np.logical_and(_lks > int(row['startframe']),
                               _lks <= int(row['endframe']))].values
                          for _, row in edf.iterrows()]

    # this variable was created to bring code into
    # line with pep8; deleting to protect against
    # changing logic
    del _lks

    edf['starttime'] = [time[edf['startframe'].iloc[ti]]
                        for ti in range(len(edf))]

    edf['endtime'] = [time[edf['endframe'].iloc[ti]]
                      for ti in range(len(edf))]

    # Proper computation of trial_length:
    # edf['trial_length'] = edf['endtime'] - edf['starttime']

    edf['change_time'] = [time[int(cf)]
                          if not np.isnan(cf)
                          else float('nan')
                          for cf in edf['change_frame']]

    edf['lick_times'] = [[time[fi] for fi in frame_arr]
                         for frame_arr in edf['lick_frames']]

    edf['trial_type'] = edf.apply(categorize_one_trial, axis=1)

    edf['reward_times'] = [[time[fi] for fi in frame_list]
                           for frame_list in edf['reward_frames']]

    edf['number_of_rewards'] = edf['reward_times'].map(len)
    edf['reward_licks'] = edf['reward_times'].apply(find_licks, args=(licks,))
    edf['reward_lick_count'] = edf['reward_licks'].map(len)

    edf['reward_lick_latency'] = edf['reward_licks'].map(lambda ll: None
                                                         if len(ll) == 0
                                                         else np.min(ll))

    # Things that dont depend on time/trial:
    edf['mouse_id'] = metadata['mouseid']
    edf['response_window'] = [metadata['response_window']] * len(edf)
    edf['task'] = metadata['task']
    edf['stage'] = metadata['stage']
    edf['session_duration'] = metadata['stoptime']
    edf['user_id'] = metadata['userid']
    edf['LDT_mode'] = metadata['lick_detect_training_mode']
    edf['blank_screen_timeout'] = metadata['blankscreen_on_timeout']
    edf['stim_duration'] = metadata['stim_duration']
    edf['blank_duration_range'] = [metadata['blank_duration_range']] * len(edf)
    edf['prechange_minimum'] = metadata['delta_minimum']
    edf['stimulus_distribution'] = metadata['stimulus_distribution']
    edf['stimulus'] = metadata['stimulus']
    edf['distribution_mean'] = metadata['delta_mean']
    edf['computer_name'] = metadata['computer_name']
    edf['behavior_session_uuid'] = metadata['behavior_session_uuid']
    edf['startdatetime'] = startdatetime
    edf['date'] = startdatetime.date()
    edf['year'] = startdatetime.year
    edf['month'] = startdatetime.month
    edf['day'] = startdatetime.day
    edf['hour'] = startdatetime.hour
    edf['dayofweek'] = startdatetime.weekday()
    edf['rig_id'] = metadata['rig_id']
    edf['cumulative_volume'] = edf['reward_volume'].cumsum()

    # Compute response latency (kinda tricky):
    edf['valid_response_licks'] = [[lk for lk in tt.lick_times
                                   if lk - tt.change_time > tt.response_window[0]]  # noqa: E50
                                   for _, tt in edf.iterrows()]

    edf['response_latency'] = edf['valid_response_licks'].map(lambda x: float('inf')  # noqa: E501
                                                              if len(x) == 0
                                                              else x[0])
    edf['response_latency'] -= edf['change_time']

    edf.drop('valid_response_licks', axis=1, inplace=True)

    # Complicated:
    assert len(edf.startdatetime.unique()) == 1
    np.testing.assert_array_equal(list(edf.index.values), np.arange(len(edf)))

    _latency = edf['response_latency'].values
    _starttime = edf['starttime'].values
    edf['reward_rate'] = calculate_reward_rate(response_latency=_latency,
                                               starttime=_starttime)

    # this variable was created to bring code into
    # line with pep8; deleting to protect against
    # changing logic
    del _latency
    del _starttime

    # Response/trial metadata encoding:
    _lt = edf['response_latency'] <= metadata['response_window'][1]
    _gt = edf['response_latency'] >= metadata['response_window'][0]
    edf['response'] = (~pd.isnull(edf['change_time']) &
                       ~pd.isnull(edf['response_latency']) &
                       _gt &
                       _lt).astype(np.float64)

    # this variable was created to bring code into
    # line with pep8; deleting to protect against
    # changing logic
    del _lt
    del _gt

    edf['response_type'] = get_response_type(edf[['trial_type',
                                                  'response',
                                                  'rewarded']])
    edf['color'] = [colormap(trial.trial_type, trial.response_type)
                    for _, trial in edf.iterrows()]

    # Reorder columns for backwards-compatibility:
    return edf[EDF_COLUMNS]


def get_extended_trials(data, time=None):
    if time is None:
        time = get_time(data)

    return create_extended_trials(trials=get_trials_v0(data, time),
                                  metadata=data_to_metadata(data, time),
                                  time=time,
                                  licks=data_to_licks(data, time))


def calculate_response_latency_list(
        trials: pd.DataFrame, response_window_start: float) -> List:
    """per trial, detemines a response latency

    Parameters
    ----------
    trials: pd.DataFrame
        contains columns "lick_times" and "change_times"
    response_window_start: float
        [seconds] relative to the non-display-lag-compensated presentation
        of the change-image

    Returns
    -------
    response_latency_list: list
        len() = trials.shape[0]
        value is 'inf' if there are no valid licks in the trial

    """
    response_latency_list = []
    for _, t in trials.iterrows():
        valid_response_licks = \
                [x for x in t.lick_times
                 if x - t.change_time > response_window_start]
        response_latency = (
                float('inf')
                if len(valid_response_licks) == 0
                else valid_response_licks[0] - t.change_time)
        response_latency_list.append(response_latency)
    return response_latency_list


def calculate_reward_rate_fix_nans(
        trials: pd.DataFrame, response_window_start: float) -> np.ndarray:
    """per trial, detemines the reward rate, replacing infs with nans

    Parameters
    ----------
    trials: pd.DataFrame
        contains columns "lick_times", "change_times", and "start_time"
    response_window_start: float
        [seconds] relative to the non-display-lag-compensated presentation
        of the change-image

    Returns
    -------
    reward_rate: np.ndarray
        size = trials.shape[0]
        value is nan if calculate_reward_rate evaluates to 'inf'

    """
    response_latency_list = calculate_response_latency_list(
            trials,
            response_window_start)
    reward_rate = calculate_reward_rate(
            response_latency=response_latency_list,
            starttime=trials.start_time.values)
    reward_rate[np.isinf(reward_rate)] = float('nan')
    return reward_rate


def construct_rolling_performance_df(trials: pd.DataFrame,
                                     response_window_start,
                                     session_type) -> pd.DataFrame:
    """Return a DataFrame containing trial by trial behavior response
    performance metrics.

    Parameters
    ----------
    trials: pd.DataFrame
        contains columns "lick_times", "change_times", and "start_time"
    response_window_start: float
        [seconds] relative to the non-display-lag-compensated presentation
        of the change-image
    session_type: str
        used to check if this was a passive session

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing:
            trials_id [index]:
                Index of the trial. All trials, including aborted trials,
                are assigned an index starting at 0 for the first trial.
            reward_rate:
                Rewards earned in the previous 25 trials, normalized by
                the elapsed time of the same 25 trials. Units are
                rewards/minute.
            hit_rate_raw:
                Fraction of go trials where the mouse licked in the
                response window, calculated over the previous 100
                non-aborted trials. Without trial count correction applied.
            hit_rate:
                Fraction of go trials where the mouse licked in the
                response window, calculated over the previous 100
                non-aborted trials. With trial count correction applied.
            false_alarm_rate_raw:
                Fraction of catch trials where the mouse licked in the
                response window, calculated over the previous 100
                non-aborted trials. Without trial count correction applied.
            false_alarm_rate:
                Fraction of catch trials where the mouse licked in
                the response window, calculated over the previous 100
                non-aborted trials. Without trial count correction applied.
            rolling_dprime:
                d prime calculated using the rolling hit_rate and
                rolling false_alarm _rate.

    """
    reward_rate = calculate_reward_rate_fix_nans(
            trials,
            response_window_start)

    # Indices to build trial metrics dataframe:
    trials_index = trials.index
    not_aborted_index = \
        trials[np.logical_not(trials.aborted)].index

    # Initialize dataframe:
    performance_metrics_df = pd.DataFrame(index=trials_index)

    # Reward rate:
    performance_metrics_df['reward_rate'] = \
        pd.Series(reward_rate, index=trials.index)

    # Hit rate raw:
    hit_rate_raw = get_hit_rate(
        hit=trials.hit,
        miss=trials.miss,
        aborted=trials.aborted)
    performance_metrics_df['hit_rate_raw'] = \
        pd.Series(hit_rate_raw, index=not_aborted_index)

    # Hit rate with trial count correction:
    hit_rate = get_trial_count_corrected_hit_rate(
            hit=trials.hit,
            miss=trials.miss,
            aborted=trials.aborted)
    performance_metrics_df['hit_rate'] = \
        pd.Series(hit_rate, index=not_aborted_index)

    # False-alarm rate raw:
    false_alarm_rate_raw = \
        get_false_alarm_rate(
                false_alarm=trials.false_alarm,
                correct_reject=trials.correct_reject,
                aborted=trials.aborted)
    performance_metrics_df['false_alarm_rate_raw'] = \
        pd.Series(false_alarm_rate_raw, index=not_aborted_index)

    # False-alarm rate with trial count correction:
    false_alarm_rate = \
        get_trial_count_corrected_false_alarm_rate(
                false_alarm=trials.false_alarm,
                correct_reject=trials.correct_reject,
                aborted=trials.aborted)
    performance_metrics_df['false_alarm_rate'] = \
        pd.Series(false_alarm_rate, index=not_aborted_index)

    # Rolling-dprime:
    if session_type.endswith('passive'):
        # It does not make sense to calculate d' for a passive session
        # So just set it to zeros
        rolling_dprime = np.zeros(len(hit_rate))
    else:
        rolling_dprime = get_rolling_dprime(hit_rate, false_alarm_rate)
    performance_metrics_df['rolling_dprime'] = \
        pd.Series(rolling_dprime, index=not_aborted_index)

    return performance_metrics_df
