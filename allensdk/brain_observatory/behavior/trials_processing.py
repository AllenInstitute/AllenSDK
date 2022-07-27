from typing import List

import pandas as pd
import numpy as np

from allensdk.brain_observatory.behavior.data_objects.trials.trial_table \
    import \
    TrialTable
from allensdk.brain_observatory.behavior.dprime import (
    get_rolling_dprime, get_trial_count_corrected_false_alarm_rate,
    get_trial_count_corrected_hit_rate,
    get_hit_rate, get_false_alarm_rate)

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
        trials: TrialTable, response_window_start: float) -> List:
    """per trial, determines a response latency

    Parameters
    ----------
    trials: TrialTable
    response_window_start: float
        [seconds] relative to the non-display-lag-compensated presentation
        of the change-image

    Returns
    -------
    response_latency_list: List
        len() = trials.shape[0]
        value is 'inf' if there are no valid licks in the trial

    Note
    -----
    response_window_start is listed as
    "relative to the non-display-lag-compensated..." because it
    comes directly from the stimulus file, which knows nothing
    about the display lag. However, response_window_start is
    only ever compared to the difference between
    trial.lick_times and trial.change_time, both of which are
    corrected for monitor delay, so it does not matter
    (the two instance of monitor delay cancel out in the
    difference).
    """
    df = pd.DataFrame({'lick_times': trials.lick_times,
                       'change_time': trials.change_time})
    df['valid_response_licks'] = df.apply(
        lambda trial: [lt for lt in trial['lick_times']
                       if lt - trial['change_time'] > response_window_start],
        axis=1)
    response_latency = df.apply(
        lambda trial: trial['valid_response_licks'][0] - trial['change_time']
        if len(trial['valid_response_licks']) > 0 else float('inf'), axis=1)
    return response_latency.tolist()


def calculate_reward_rate_fix_nans(
        trials: TrialTable, response_window_start: float) -> np.ndarray:
    """per trial, determines the reward rate, replacing infs with nans

    Parameters
    ----------
    trials: TrialTable
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


def construct_rolling_performance_df(trials: TrialTable,
                                     response_window_start,
                                     session_type) -> pd.DataFrame:
    """Return a DataFrame containing trial by trial behavior response
    performance metrics.

    Parameters
    ----------
    trials: TrialTable
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
        trials.value[np.logical_not(trials.aborted)].index

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
