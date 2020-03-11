import numpy as np
import datetime
import pandas as pd
import scipy.stats as sps
import uuid
from copy import deepcopy
import collections
import dateutil
from scipy.stats import norm

from allensdk import one


# TODO: add trial column descriptions
TRIAL_COLUMN_DESCRIPTION_DICT = {}

EDF_COLUMNS = ['index', 'lick_times', 'auto_rewarded', 'cumulative_volume',
       'cumulative_reward_number', 'reward_volume', 'reward_times',
       'reward_frames', 'rewarded', 'optogenetics', 'response_type',
       'response_time', 'change_time', 'change_frame', 'response_latency',
       'starttime', 'startframe', 'trial_length', 'scheduled_change_time',
       'endtime', 'endframe', 'initial_image_category', 'initial_image_name',
       'change_image_name', 'change_image_category', 'change_ori',
       'change_contrast', 'initial_ori', 'initial_contrast', 'delta_ori',
       'mouse_id', 'response_window', 'task', 'stage', 'session_duration',
       'user_id', 'LDT_mode', 'blank_screen_timeout', 'stim_duration',
       'blank_duration_range', 'prechange_minimum', 'stimulus_distribution',
       'stimulus', 'distribution_mean', 'computer_name',
       'behavior_session_uuid', 'startdatetime', 'date', 'year', 'month',
       'day', 'hour', 'dayofweek', 'number_of_rewards', 'rig_id', 'trial_type',
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
                initial_image_group = initial_image_name = set_event[1]  # hack assumes initial_image_group == initial_image_name, only initial_image_name is present for natual_scenes
                initial_image_category_name = stim_category_name
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
    * sham_change: stimulus did not change, but response was evaluated (True on 'catch' trials)

    Each trial can be one (and only one) of the following:
    * hit (stimulus changed, animal responded in response window)
    * miss (stimulus changed, animal did not respond in response window)
    * false_alarm (stimulus did not change, animal responded in response window)
    * correct_reject (stimulus did not change, animal did not respond in response window)
    * aborted (animal responded before change time)
    * auto_rewarded (reward was automatically delivered following the change. This will bias the animals choice and should not be categorized as hit/miss)
    
    
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
    '''ensure that only one of N possible mutually exclusive trial conditions is True'''
    on = []
    for condition, value in trial_conditions.items():
        if value:
            on.append(condition)
    
    if len(on) != 1:
        all_conditions = list(trial_conditions.keys())
        raise AssertionError(f"expected exactly 1 trial condition out of {all_conditions} to be True, instead {on} were True (trial {trial_index})")


def get_trial_lick_times(lick_times, start_time, stop_time):
    '''extract lick times in time range'''
    return lick_times[np.where(np.logical_and(
        lick_times >= start_time, 
        lick_times <= stop_time
    ))]


def get_trial_reward_time(rebased_reward_times, start_time, stop_time):
    '''extract reward times in time range'''
    reward_times = rebased_reward_times[np.where(np.logical_and(
        rebased_reward_times >= start_time, 
        rebased_reward_times <= stop_time
    ))]
    return float('nan') if len(reward_times) == 0 else one(reward_times)
        

def get_trial_timing(event_dict, stimulus_presentations_df, licks, go, catch, auto_rewarded, hit, false_alarm):
    '''
    extract trial timing data
    
    content of trial log depends on trial type depends on trial type and response type
    go, catch, auto_rewarded, hit, false_alarm must be passed as booleans to disambiguate trial and response type

    on `go` or `auto_rewarded` trials, extract the stimulus_changed time
    on `catch` trials, extract the sham_change time

    on `hit` trials, extract the response time from the `hit` entry in event_dict
    on `false_alarm` trials, extract the response time from the `false_alarm` entry in event_dict
    '''

    assert not (hit==True and false_alarm==True), "both `hit` and `false_alarm` cannot be True, they are mutually exclusive categories"
    assert not (go==True and catch==True), "both `go` and `catch` cannot be True, they are mutually exclusive categories"
    assert not (go==True and auto_rewarded==True), "both `go` and `auto_rewarded` cannot be True, they are mutually exclusive categories"

    start_time = event_dict["trial_start", ""]['rebased_time']
    stop_time = event_dict["trial_end", ""]['rebased_time']

    if hit:
        response_time = event_dict.get(("hit", ""))['rebased_time']
    elif false_alarm:
        response_time = event_dict.get(("false_alarm", ""))['rebased_time']
    else:
        response_time = float("nan")

    def get_change_time(change_frame,stimulus_presentations_df):
        # get the first stimulus in the log after the current change frame:
        query = stimulus_presentations_df.query('start_frame >= @change_frame')
        if len(query) > 0:
            return query['start_time'].iloc[0]
        else:
            # return NaN if the query is empty
            return np.nan

    if go or auto_rewarded:
        change_frame = event_dict.get(('stimulus_changed', ''))['frame']
        change_time = get_change_time(change_frame,stimulus_presentations_df)
    elif catch:
        change_frame = event_dict.get(('sham_change', ''))['frame']
        change_time = get_change_time(change_frame,stimulus_presentations_df)
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


def get_trial_image_names(trial, stimuli):
    trial_start_frame = trial["events"][0][3]
    _, _, initial_image_name = resolve_initial_image(stimuli, trial_start_frame)
    if len(trial["stimulus_changes"]) == 0:
        change_image_name = initial_image_name
    else:
        (_, from_name), (_, to_name), _, _ = trial["stimulus_changes"][0]
        assert from_name == initial_image_name
        change_image_name = to_name

    return {
        "initial_image_name": initial_image_name,
        "change_image_name": change_image_name
    }


def get_trials(data, licks_df, rewards_df, stimulus_presentations_df, rebase):
    assert rewards_df.index.name == 'timestamps'
    stimuli = data["items"]["behavior"]["stimuli"]
    trial_log = data["items"]["behavior"]["trial_log"]

    all_trial_data = [None] * len(trial_log)
    sync_lick_times = licks_df.time.values 
    rebased_reward_times = rewards_df.index.values

    for idx, trial in enumerate(trial_log):
        # extract rebased time and frame for each event in the trial log:
        event_dict = {(e[0], e[1]): {'rebased_time':rebase(e[2]),'frame':e[3]} for e in trial['events']}

        tr_data = {"trial": trial["index"]}
        tr_data["lick_times"] = get_trial_lick_times(
            sync_lick_times, 
            event_dict[('trial_start', '')]['rebased_time'], 
            event_dict[('trial_end', '')]['rebased_time']
        )
        tr_data["reward_time"] = get_trial_reward_time(
            rebased_reward_times,
            event_dict[('trial_start', '')]['rebased_time'], 
            event_dict[('trial_end', '')]['rebased_time']
        )
        tr_data.update(trial_data_from_log(trial))
        tr_data.update(get_trial_timing(
            event_dict,
            stimulus_presentations_df,
            tr_data['lick_times'],
            tr_data['go'],
            tr_data['catch'],
            tr_data['auto_rewarded'],
            tr_data['hit'],
            tr_data['false_alarm'],
        ))
        tr_data.update(get_trial_image_names(trial, stimuli))

        # ensure that only one trial condition is True (they are mutually exclusive)
        condition_dict = {}
        for key in ['hit','miss','false_alarm','correct_reject','auto_rewarded','aborted']:
            condition_dict[key] = tr_data[key]
        validate_trial_condition_exclusivity(idx,**condition_dict)

        all_trial_data[idx] = tr_data

    trials = pd.DataFrame(all_trial_data).set_index('trial')
    trials.index = trials.index.rename('trials_id')
    del trials["sham_change"]

    return trials


def local_time(iso_timestamp, timezone=None):
    datetime = pd.to_datetime(iso_timestamp)
    if not datetime.tzinfo:
        datetime = datetime.replace(tzinfo=dateutil.tz.gettz('America/Los_Angeles'))
    return datetime.isoformat()


def get_time(exp_data):
    vsyncs = exp_data["items"]["behavior"]["intervalsms"]
    return np.hstack((0, vsyncs)).cumsum() / 1000.0


def data_to_licks(data, time):
    lick_frames = data['items']['behavior']['lick_sensors'][0]['lick_events']
    lick_times = time[lick_frames]
    return pd.DataFrame(data={"frame": lick_frames, 'time': lick_times})


def get_mouse_id(exp_data):
    return exp_data["items"]["behavior"]['config']['behavior']['mouse_id']


def get_params(exp_data):

    params = deepcopy(exp_data["items"]["behavior"].get("params", {}))
    params.update(exp_data["items"]["behavior"].get("cl_params", {}))

    if "response_window" in params:
        params["response_window"] = list(params["response_window"])  # tuple to list

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
        if stim['obj_type'].lower() == 'docimagestimulus' and stim['sampling'] in ['even', 'file']:
            return True
    return False


def data_to_metadata(data, time):

    metadata = {
        "startdatetime": local_time(data["start_time"], timezone='America/Los_Angeles'),
        "rig_id": RIG_NAME.get(data['platform_info']['computer_name'].lower(), 'unknown'),
        "computer_name": data['platform_info']['computer_name'],
        "reward_vol": data["items"]["behavior"]["config"]["reward"]["reward_volume"],
        "auto_reward_vol": data["items"]["behavior"]["config"]["DoC"]["auto_reward_volume"],
        "params": get_params(data),
        "mouseid": data["items"]["behavior"]['config']['behavior']['mouse_id'],
        "response_window": list(data["items"]["behavior"].get("config", {}).get("DoC", {}).get("response_window")),
        "task": data["items"]["behavior"]["config"]["behavior"]['task_id'],
        "stage": data["items"]["behavior"]["params"]["stage"],
        "stoptime": time[-1] - time[0],
        "userid": data["items"]["behavior"]['cl_params']['user_id'],
        "lick_detect_training_mode": "single",
        "blankscreen_on_timeout": False,
        "stim_duration": data["items"]["behavior"]["config"]['DoC']['stimulus_window'] * 1000,
        "blank_duration_range": list(data["items"]["behavior"]["config"]['DoC']['blank_duration_range']),
        "delta_minimum": data["items"]["behavior"]["config"]['DoC']['pre_change_time'],
        "stimulus_distribution": data["items"]["behavior"]["config"]["DoC"]["change_time_dist"],
        "delta_mean": data["items"]["behavior"]['config']["DoC"]["change_time_scale"],
        "trial_duration": None,
        "n_stimulus_frames": sum([sum(s.get("draw_log", [])) for s in data["items"]["behavior"]["stimuli"].values()]),
        "stimulus": list(data["items"]["behavior"]["stimuli"].keys())[0],
        "warm_up_trials": data["items"]["behavior"]["config"]["DoC"]["warm_up_trials"],
        "stimulus_window": data["items"]["behavior"]["config"]["DoC"]["stimulus_window"],
        "volume_limit": data["items"]["behavior"]["config"]["behavior"]["volume_limit"],
        "failure_repeats": data["items"]["behavior"]["config"]["DoC"]["failure_repeats"],
        "catch_frequency": data["items"]["behavior"]["config"]["DoC"]["catch_freq"],
        "auto_reward_delay": data["items"]["behavior"]["config"]["DoC"].get("auto_reward_delay", 0.0),
        "free_reward_trials": data["items"]["behavior"]["config"]["DoC"]["free_reward_trials"],
        "min_no_lick_time": data["items"]["behavior"]["config"]["DoC"]["min_no_lick_time"],
        "max_session_duration": data["items"]["behavior"]["config"]["DoC"]["max_task_duration_min"],
        "abort_on_early_response": data["items"]["behavior"]["config"]["DoC"]["abort_on_early_response"],
        "initial_blank_duration": data["items"]["behavior"]["config"]["DoC"]["initial_blank"],
        "even_sampling_enabled": get_even_sampling(data),
        "behavior_session_uuid": uuid.UUID(data["session_uuid"]),
        "periodic_flash": data['items']['behavior']['config']['DoC']['periodic_flash'],
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
            return change_event[2], change_event[3], get_response_latency(change_event, trial)
    return None, None, None

def get_stimulus_attr_changes(stim_dict, change_frame, first_frame, last_frame):
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
        (from_group, from_name, ), (to_group, to_name), _, _ = trial_log[ti]["stimulus_changes"][0]
        return from_group, from_name, to_group, to_name
    else:
        _, _, prev_group, prev_name = get_image_info_from_trial(trial_log, ti - 1)
        return prev_group, prev_name, prev_group, prev_name


def get_ori_info_from_trial(trial_log, ti, ):
    if ti == -1:
        raise IndexError('No change on first trial.')
    
    if len(trial_log[ti]["stimulus_changes"]) == 1:
        (initial_group, initial_orientation), (change_group, change_orientation, ), _, _ = trial_log[ti]["stimulus_changes"][0]
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

    implied_type = stim["obj_type"]
    trial_log = data["items"]["behavior"]["trial_log"]
    pre_change_time = data["items"]["behavior"]["config"]['DoC']['pre_change_time']
    initial_blank_duration = data["items"]["behavior"]["config"]["DoC"]["initial_blank"]

    initial_stim = stim['set_log'][0]  # we need this for the situations where a change doesn't occur on the first trial

    trials = collections.defaultdict(list)
    for ti, trial in enumerate(trial_log):

        trials['index'].append(trial["index"])
        trials['lick_times'].append([lick[0] for lick in trial["licks"]])
        trials['auto_rewarded'].append(trial["trial_params"]["auto_reward"] if trial['trial_params']['catch'] == False else None)
        trials['cumulative_volume'].append(trial["cumulative_volume"])
        trials['cumulative_reward_number'].append(trial["cumulative_rewards"])
        trials['reward_volume'].append(sum([r[0] for r in trial.get("rewards", [])]))
        trials['reward_times'].append([reward[1] for reward in trial["rewards"]])
        trials['reward_frames'].append([reward[2] for reward in trial["rewards"]])
        trials['rewarded'].append(trial["trial_params"]["catch"] is False)
        trials['optogenetics'].append(trial["trial_params"].get("optogenetics", False))
        trials['response_type'].append([])
        trials['response_time'].append([])
        trials['change_time'].append(get_change_time_frame_response_latency(trial)[0])
        trials['change_frame'].append(get_change_time_frame_response_latency(trial)[1])
        trials['response_latency'].append(get_change_time_frame_response_latency(trial)[2])
        trials['starttime'].append(trial["events"][0][2])
        trials['startframe'].append(trial["events"][0][3])
        trials['trial_length'].append(trial["events"][-1][2] - trial["events"][0][2])
        trials['scheduled_change_time'].append(pre_change_time + initial_blank_duration + trial["trial_params"]["change_time"])
        trials['endtime'].append(trial["events"][-1][2])
        trials['endframe'].append(trial["events"][-1][3])

        # Stimulus:
        if implied_type == 'DoCImageStimulus':
            from_group, from_name, to_group, to_name = get_image_info_from_trial(trial_log, ti)
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
                change_orientation, initial_orientation, delta_orientation = get_ori_info_from_trial(trial_log, ti)
            except IndexError:
                orientation = initial_stim[1]  # shape: group_name, orientation, stimulus time relative to start, frame
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
            raise NotImplementedError('Unsupported stimulus type: {}'.format(implied_type), )

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
        reward_lick_mask = ((licks['time'] > reward_time) & (licks['time'] < (reward_time + window)))

        tr_licks = licks[reward_lick_mask].copy()
        tr_licks['time'] -= reward_time
        return tr_licks['time'].values


def calculate_reward_rate(response_latency=None, starttime=None, window=0.75, trial_window=25, initial_trials=10):
    assert len(response_latency) == len(starttime)

    df = pd.DataFrame({'response_latency': response_latency, 'starttime':starttime})

    # adds a column called reward_rate to the input dataframe
    # the reward_rate column contains a rolling average of rewards/min
    # window sets the window in which a response is considered correct, so a window of 1.0 means licks before 1.0 second are considered correct
    # Reorganized into this unit-testable form by Nick Cain April 25 2019

    reward_rate = np.zeros(len(df))
    reward_rate[:initial_trials] = np.inf  # make the initial reward rate infinite, so that you include the first trials automatically.
    for trial_number in range(initial_trials, len(df)):

        min_index = np.max((0, trial_number - trial_window))
        max_index = np.min((trial_number + trial_window, len(df)))
        df_roll = df.iloc[min_index:max_index]

        correct = len(df_roll[df_roll.response_latency < window])  # get a rolling number of correct trials
        time_elapsed = df_roll.starttime.iloc[-1] - df_roll.starttime.iloc[0]  # get the time elapsed over the trials
        reward_rate_on_this_lap = correct / time_elapsed * 60 # calculate the reward rate, rewards/min

        reward_rate[trial_number] = reward_rate_on_this_lap
    return reward_rate


def get_response_type(trials):

    response_type = []
    for idx in trials.index:
        if trials.loc[idx].trial_type.lower() == 'aborted':
            response_type.append('EARLY_RESPONSE')
        elif (trials.loc[idx].rewarded == True) & (trials.loc[idx].response == 1):
            response_type.append('HIT')
        elif (trials.loc[idx].rewarded == True) & (trials.loc[idx].response != 1):
            response_type.append('MISS')
        elif (trials.loc[idx].rewarded == False) & (trials.loc[idx].response == 1):
            response_type.append('FA')
        elif (trials.loc[idx].rewarded == False) & (trials.loc[idx].response != 1):
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
    edf = trials[~pd.isnull(trials['reward_times'])].reset_index(drop=True).copy()

    # Buggy computation of trial_length (for backwards compatibility)
    edf.drop(['trial_length'], axis=1, inplace=True)
    edf['endtime_buggy'] = [edf['starttime'].iloc[ti + 1] if ti < len(edf) - 1 else time[-1] for ti in range(len(edf))]
    edf['trial_length'] = edf['endtime_buggy'] - edf['starttime']
    edf.drop(['endtime_buggy'], axis=1, inplace=True)

    # Make trials contiguous, and rebase time:
    edf.drop(['endframe', 'starttime', 'endtime', 'change_time', 'lick_times', 'reward_times'], axis=1, inplace=True)
    edf['endframe'] = [edf['startframe'].iloc[ti + 1] if ti < len(edf) - 1 else len(time) - 1 for ti in range(len(edf))]
    edf['lick_frames'] = [licks['frame'][np.logical_and(licks['frame'] > int(row['startframe']), licks['frame'] <= int(row['endframe']))].values for _, row in edf.iterrows()]
    edf['starttime'] = [time[edf['startframe'].iloc[ti]] for ti in range(len(edf))]
    edf['endtime'] = [time[edf['endframe'].iloc[ti]] for ti in range(len(edf))]
    
    # Proper computation of trial_length:
    # edf['trial_length'] = edf['endtime'] - edf['starttime']

    edf['change_time'] = [time[int(cf)] if not np.isnan(cf) else float('nan') for cf in edf['change_frame']]
    edf['lick_times'] = [[time[fi] for fi in frame_arr] for frame_arr in edf['lick_frames']]
    edf['trial_type'] = edf.apply(categorize_one_trial, axis=1)
    edf['reward_times'] = [[time[fi] for fi in frame_list] for frame_list in edf['reward_frames']]
    edf['number_of_rewards'] = edf['reward_times'].map(len)
    edf['reward_licks'] = edf['reward_times'].apply(find_licks, args=(licks,))
    edf['reward_lick_count'] = edf['reward_licks'].map(len)
    edf['reward_lick_latency'] = edf['reward_licks'].map(lambda ll: None if len(ll) == 0 else np.min(ll))

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
    edf['valid_response_licks'] = [[l for l in t.lick_times if l - t.change_time > t.response_window[0]] for _, t in edf.iterrows()]
    edf['response_latency'] = edf['valid_response_licks'].map(lambda x: float('inf') if len(x) == 0 else x[0]) - edf['change_time']
    edf.drop('valid_response_licks', axis=1, inplace=True)

    # Complicated:
    assert len(edf.startdatetime.unique()) == 1
    np.testing.assert_array_equal(list(edf.index.values), np.arange(len(edf)))
    edf['reward_rate'] = calculate_reward_rate(response_latency=edf['response_latency'].values, starttime=edf['starttime'].values)

    # Response/trial metadata encoding:
    edf['response'] = (~pd.isnull(edf['change_time']) &
                       ~pd.isnull(edf['response_latency']) &
                       (edf['response_latency'] >= metadata['response_window'][0]) &
                       (edf['response_latency'] <= metadata['response_window'][1])).astype(np.float64)
    edf['response_type'] = get_response_type(edf[['trial_type', 'response', 'rewarded']])
    edf['color'] = [colormap(trial.trial_type, trial.response_type) for _, trial in edf.iterrows()]

    # Reorder columns for backwards-compatibility:
    return edf[EDF_COLUMNS]

def get_extended_trials(data, time=None):
    if time is None:
        time = get_time(data)

    return create_extended_trials(trials=get_trials_v0(data, time),
                                  metadata=data_to_metadata(data, time),
                                  time=time,
                                  licks=data_to_licks(data, time))


