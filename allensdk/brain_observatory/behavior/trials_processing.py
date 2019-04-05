import numpy as np
import datetime
import pandas as pd
import scipy.stats as sps

from dateutil import parser
from six import iteritems
from collections import defaultdict

TRIAL_COLUMN_DESCRIPTION_DICT = {}

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

    for stim_category_name, stim_dict in iteritems(stimuli):
        for set_event in stim_dict["set_log"]:
            set_frame = set_event[3]
            if set_frame <= start_frame and set_frame >= max_frame:
                initial_image_group = initial_image_name = set_event[1]  # hack assumes initial_image_group == initial_image_name, only initial_image_name is present for natual_scenes
                initial_image_category_name = stim_category_name
                max_frame = set_frame

    return initial_image_category_name, initial_image_group, initial_image_name


def get_trials(data, stimulus_timestamps_no_monitor_delay, licks_df, rewards_df, rebase):
    assert rewards_df.index.name == 'timestamps'
    stimuli = data["items"]["behavior"]["stimuli"]

    trial_data = defaultdict(list)
    sync_lick_times = licks_df.time.values 
    rebased_reward_times = rewards_df.index.values
    for trial in data["items"]["behavior"]["trial_log"]:
        event_dict = {(e[0], e[1]): rebase(e[2]) for e in trial['events']}

        trial_data['trial'].append(trial["index"])

        start_time = event_dict['trial_start', '']
        trial_data['start_time'].append(start_time)

        stop_time = event_dict['trial_end', '']
        trial_data['stop_time'].append(stop_time)

        trial_length = stop_time - start_time
        trial_data['trial_length'].append(trial_length)

        catch = trial["trial_params"]["catch"] == True
        trial_data['catch'].append(catch)

        auto_rewarded = trial["trial_params"]["auto_reward"]
        trial_data['auto_rewarded'].append(auto_rewarded)

        go = not catch and not auto_rewarded
        trial_data['go'].append(go)

        lick_events = [rebase(lick_tuple[0]) for lick_tuple in trial["licks"]]
        trial_data['lick_events'].append(lick_events)

        lick_times = sync_lick_times[np.where(np.logical_and(sync_lick_times >= start_time, sync_lick_times <= stop_time))]
        trial_data['lick_times'].append(lick_times)

        aborted = ("abort", "") in event_dict
        trial_data['aborted'].append(aborted)

        reward_volume = sum([r[0] for r in trial.get("rewards", [])])
        trial_data['reward_volume'].append(reward_volume)

        hit = ('hit', "") in event_dict
        trial_data['hit'].append(hit)

        false_alarm = ('false_alarm', "") in event_dict
        trial_data['false_alarm'].append(false_alarm)

        response_time = event_dict.get(('hit', '')) or event_dict.get(('false_alarm', '')) if hit or false_alarm else None
        trial_data['response_time'].append(response_time)

        miss = ('miss', "") in event_dict
        trial_data['miss'].append(miss)

        reward_times = rebased_reward_times[np.where(np.logical_and(rebased_reward_times >= start_time, rebased_reward_times <= stop_time))]
        trial_data['reward_times'].append(reward_times)

        sham_change = True if ('sham_change', '') in event_dict else False
        trial_data['sham_change'].append(sham_change)

        stimulus_change = True if ('stimulus_changed', '') in event_dict else False
        trial_data['stimulus_change'].append(stimulus_change)

        change_time = event_dict.get(('stimulus_changed', '')) or event_dict.get(('sham_change', '')) if stimulus_change or sham_change else None
        trial_data['change_time'].append(change_time)

        if not (sham_change or stimulus_change):
            response_latency = None
        else:
            if hit or false_alarm:
                response_latency = response_time - change_time
            else:
                response_latency = float("inf")
        trial_data['response_latency'].append(response_latency)

        trial_start_frame = trial["events"][0][3]
        _, _, initial_image_name = resolve_initial_image(stimuli, trial_start_frame)
        if len(trial["stimulus_changes"]) == 0:
            change_image_name = initial_image_name
        else:
            (_, from_name), (_, to_name), _, _ = trial["stimulus_changes"][0]
            assert from_name == initial_image_name
            change_image_name = to_name
        trial_data['initial_image_name'].append(initial_image_name)
        trial_data['change_image_name'].append(change_image_name)

    trials = pd.DataFrame(trial_data).set_index('trial')
    trials.index = trials.index.rename('trials_id')

    return trials
