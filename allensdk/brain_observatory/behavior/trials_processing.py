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
