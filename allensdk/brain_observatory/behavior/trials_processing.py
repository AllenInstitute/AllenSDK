import numpy as np
import pandas as pd
from six import iteritems
from collections import defaultdict

    # stimuli = data["items"]["behavior"]["stimuli"]

    # pre_change_time = data["items"]["behavior"]["config"]["DoC"]["pre_change_time"]     #    .get("pre_change_time", 0)
    # initial_blank_duration = data["items"]["behavior"]["config"]["DoC"]["initial_blank"] or 0  # woohoo!

    # annotate_optogenetics = lambda trial: {"optogenetics": trial["trial_params"].get("optogenetics", False)}

def _resolve_initial_image(stimuli, start_frame):
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
                # initial_image_group, initial_image_name = change_event[0]
                initial_image_category_name = stim_category_name
                max_frame = set_frame

    return initial_image_category_name, initial_image_group, initial_image_name


def _resolve_stimulus_dict(stimuli, group_name):
    """
    Notes
    -----
    - will return the first stimulus dict with group_name present in its list of
    stim_groups
    """
    for classification_name, stim_dict in iteritems(stimuli):
        if group_name in stim_dict["stim_groups"]:
            return classification_name, stim_dict
    else:
        raise ValueError("unable to resolve stimulus_dict from group_name...")

def annotate_rewards(trial):
    """Annotate rewards-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        auto_rewarded_trial: bool
            whether or not the trial is autorewarded
        cumulative_volume: float
            cumulative water volume dispensed relative to start of experiment
        cumulative_reward_number: int
            cumulative reward number dispensed relative to start of experiment
        reward_times: list of float
            times (s) in which rewards occurred
        reward_frames: list of int
            frame indices in which rewards occurred
        rewarded: bool
            whether or not it's a rewarded trial

    Notes
    -----
    - time is seconds since start of experiment
    """
    return {
        "auto_rewarded_trial": trial["trial_params"]["auto_reward"] if trial['trial_params']['catch'] == False else None,
        "cumulative_volume": trial["cumulative_volume"],
        "cumulative_reward_number": trial["cumulative_rewards"],
        "reward_volume": sum([r[0] for r in trial.get("rewards", [])]),
        "reward_times": [reward[1] for reward in trial["rewards"]],
        "reward_frames": [reward[2] for reward in trial["rewards"]],
        "rewarded": trial["trial_params"]["catch"] == False,  # justink said: assume go trials are catch != True, assume go trials are the only type of rewarded trials
    }

def annotate_schedule_time(trial, pre_change_time, initial_blank_duration):
    """Annotate time/scheduling-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log
    pre_change_time: float
        minimum time before a stimulus change occurs in the trial
    initial_blank_duration: float
        time at start before the prechange window

    Returns
    -------
    dict
        start_time: float
            time (s) of the start of the trial
        start_frame: int
            frame index of the start of the trial
        scheduled_change_time: float
            time (s) scheduled for the change to occur

    Notes
    -----
    - time is seconds since start of experiment
    """
    try:
        start_time, start_frame = trial["events"][0][2:4]
        end_time, end_frame = trial["events"][-1][2:4]
        trial_length = end_time - start_time
    except IndexError:
        return {
            "start_time": None,
            "start_frame": None,
            "trial_length": None,
            "scheduled_change_time": None,
            "end_time": None,
            "end_frame": None,
        }

    return {
        "start_time": start_time,
        "start_frame": start_frame,
        "trial_length": trial_length,
        "scheduled_change_time": (
            pre_change_time +
            initial_blank_duration +
            trial["trial_params"]["change_time"]
        ),  # adding start time will put this in time relative to start of exp, change_time is relative to after prechange + initial_blank_duration, using () because of annoying flake8 bug + noqa directive not working for multiline
        "end_time": end_time,
        "end_frame": end_frame,
    }


def annotate_stimuli(trial, stimuli):
    """Annotate stimuli-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log
    stimuli: Mapping
        foraging2 shape stimuli mapping

    Returns
    -------
    dict
        initial_image_name: str
            name of the initial image
        initial_image_category: str
            category of the initial image
        change_image_name: str
            name of the change image
        change_image_category: str
            category of the change image
        change_orientation: float
            the orientation of the change image
        change_contrast: float
            the contrast of the change image
        initial_orientation: float
            the orientation of the initial image
        initial_contrast: float
            the contrast of the initial image
        delta_orientation: float
            difference in orientation between the initial image and the change image

    Notes
    -----
    - time is seconds since start of experiment
    - the initial image is the image shown before the "change event" occurs
    - the change image is the image shown after the "change event" occurs
    - assumes only one stimulus change can occur in a single trial and that
    each change will only be intra-classification (ie: "natural_images",
    "gratings", etc.)
    - currently only supports the following stimuli types and none of their
    subclasses: DoCGratingStimulus, DoCImageStimulus
    - if you are mixing more than one classification of stimulus, it will
    resolve the first group_name, stimulus_name pair it encounters...so maybe
    name things uniquely everywhere...
    """
    try:
        stimulus_change = trial["stimulus_changes"][0]
        (from_group, from_name, ), (to_group, to_name), _, change_frame = stimulus_change
        _, stim_dict = _resolve_stimulus_dict(stimuli, from_group)
    except IndexError:
        trial_start_frame = trial["events"][0][3]
        category, from_group, from_name = _resolve_initial_image(
            stimuli,
            trial_start_frame
        )

        to_group, to_name = from_group, from_name
        change_frame = np.nan

        if category:
            stim_dict = stimuli[category]
        else:
            stim_dict = {}

    implied_type = stim_dict["obj_type"]

    # initial is like ori before, contrast before...etc
    # type is image or grating, changes is a dictionary of changes make sure each change type is lowerccase string...

    if implied_type in ("DoCGratingStimulus", ):
        first_frame, last_frame = _get_trial_frame_bounds(trial)

        initial_changes, change_changes = _get_stimulus_attr_changes(
            stim_dict, change_frame, first_frame, last_frame
        )

        initial_orientation = initial_changes.get("ori")
        initial_contrast = initial_changes.get("contrast")
        change_orientation = change_changes.get("ori", initial_orientation)
        change_contrast = change_changes.get("constrast", initial_contrast)

        if initial_orientation is not None and change_orientation is not None:
            delta_orientation = change_orientation - initial_orientation
        else:
            delta_orientation = np.nan

        return {
            "initial_image_category": '',
            "initial_image_name": '',
            "change_image_name": '',
            "change_image_category": '',
            "change_orientation": change_orientation,
            "change_contrast": change_contrast,
            "initial_orientation": initial_orientation,
            "initial_contrast": initial_contrast,
            "delta_orientation": delta_orientation,
        }
    elif implied_type in ("DoCImageStimulus", ):
        return {
            "initial_image_category": from_group,
            "initial_image_name": from_name,
            "change_image_name": to_name,
            "change_image_category": to_group,
            "change_orientation": None,
            "change_contrast": None,
            "initial_orientation": None,
            "initial_contrast": None,
            "delta_orientation": None,
        }
    else:
        raise ValueError("invalid implied type: {}".format(implied_type))

def annotate_responses(trial):
    """Annotate response-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        response_time: float
            time (s) of the response
        response_type: str
            type of response
        response_latency: float, np.nan, np.inf
            difference between stimulus change and response, np.nan if a change doesn't
            occur, np.inf if a change occurs but a response doesn't occur
        rewarded: boolean, None


    Notes
    -----
    - time is seconds since start of experiment
    - 03/13/18: justin did not know what to use for `response_type` key so it will be None
    """

    for event in trial['events']:
        if event[0] == 'stimulus_changed':
            change_event = event
            break
        elif event[0] == 'sham_change':
            change_event = event
            break
        else:
            change_event = None

    for event in trial['events']:
        if event[0] == 'hit':
            response_event = event
            break
        elif event[0] == 'false_alarm':
            response_event = event
            break
        else:
            response_event = None

    if change_event is None:
        # aborted
        return {
            # "response_frame": frame,
            "response_type": [],
            "response_time": [],
            "response_latency": None,
            "change_frame": None,
            "change_time": None,
        }

    elif change_event[0] == 'stimulus_changed':
        if response_event is None:
            # miss
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": np.inf,
            }
        elif response_event[0] == 'hit':
            # hit
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": response_event[2] - change_event[2],
            }

        else:
            raise Exception('something went wrong')
    elif change_event[0] == 'sham_change':
        if response_event is None:
            # correct reject
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": np.inf,
            }
        elif response_event[0] == 'false_alarm':
            # false alarm
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": response_event[2] - change_event[2],
            }
        else:
            raise Exception('something went wrong')
    else:
        raise Exception('something went wrong')

# def expand_dict(out_dict, from_dict, index):
#     """there is obviously a better way...

#     Notes
#     -----
#     - TODO replace this by making every `annotate_...` function return a pandas Series
#     """
#     for k, v in from_dict.items():
#         if not out_dict.get(k):
#             out_dict[k] = {}

#         out_dict.get(k)[index] = v

import scipy.stats as sps

def get_trials(data, stimulus_timestamps_sync, licks, rewards, sync_lick_times):

    # Time rebasing: times in stimulus_timestamps_pickle and lick log will agree with times in event log
    vsyncs = data["items"]["behavior"]['intervalsms']
    stimulus_timestamps_pickle_pre = np.hstack((0, vsyncs)).cumsum() / 1000.0
    assert len(stimulus_timestamps_pickle_pre) == len(stimulus_timestamps_sync)
    first_trial = data["items"]["behavior"]["trial_log"][0]
    first_trial_start_time, first_trial_start_frame = {(e[0], e[1]):(e[2], e[3]) for e in first_trial['events']}['trial_start','']
    offset_time = first_trial_start_time-stimulus_timestamps_pickle_pre[first_trial_start_frame]
    stimulus_timestamps_pickle = np.array([t+offset_time for t in stimulus_timestamps_pickle_pre])
    lick_log_times_pre = np.array([offset_time+x for x in licks['time'].values])

    # Rebase used to transform trial log times to sync times:
    time_slope, time_intercept, _, _, _ = sps.linregress(stimulus_timestamps_pickle, stimulus_timestamps_sync)
    def rebase(t):
        return time_intercept+time_slope*t
    lick_log_times = np.array([rebase(x) for x in lick_log_times_pre])


    event_set = set()
    trial_data = defaultdict(list)
    licks_log_counter = 0
    all_licks = []
    hwm = float("-inf")
    for trial in data["items"]["behavior"]["trial_log"]:
        event_dict = {(e[0], e[1]):rebase(e[2]) for e in trial['events']}
        
        trial_data['trial'].append(trial["index"])

        start_time = event_dict['trial_start', '']
        trial_data['start_time'].append(start_time)

        end_time = event_dict['trial_end', '']
        trial_data['end_time'].append(end_time)
        
        curr_trial_lick_times = sync_lick_times[np.where(np.logical_and(sync_lick_times >= start_time, sync_lick_times <= end_time))]
        trial_data['lick_times'].append(curr_trial_lick_times)


        # Use to debug monitor delay:
        # LL1, LL2 = [l for l in curr_licks_from_lick_events], [rebase(lick_tuple[0]) for lick_tuple in trial["licks"]]
        # if len(LL1) == len(LL2):
        #     print [x-y for x, y in zip(LL1, LL2)]

    
    # print hwm
        # for lt in LL2:
        #     xi = np.where(stimulus_timestamps_pickle>=lt)[0][0]
        #     # print xi[0][0]
        #     print lt - stimulus_timestamps_pickle[xi-1]
        # print
        # print LL1
        # print LL2

        # event_list = []
        # for e in trial['events']:
        #     event_list.append(tuple(e))
        #     event_set.add((event_list[-1][:2]))

        # for lick_tuple in trial["licks"]:
        #     licks_log_counter += 1
            # event_list.append(tuple(['lick', '', lick_tuple[0], lick_tuple[1]]))
            # event_set.add((event_list[-1][:2]))

        # print
        # for e in sorted(event_list, key=lambda x:x[2]):
        #     print e
            
    # print 'licks_log:', licks_log_counter


    # missing_times = []
    # for lt in lick_log_times:

    #     if lt not in all_licks_from_lick_events:
    #         missing_times.append(lt)
    #         print lt

    # first_trial = data["items"]["behavior"]["trial_log"][0]
    # first_trial_start = {(e[0], e[1]):(e[2], e[3]) for e in first_trial['events']}['trial_start',''][0]
    # print rebase(first_trial_start)
    #     # if offset_time is None:
    #     #     ts, fs = event_dict

    # print sync_lick_times[np.where(sync_lick_times<missing_times[0])]
    # print sync_lick_times[np.where(sync_lick_times>missing_times[-1])]



        # trial_data['lick_times'].append([rebase(lick_tuple[0]) for lick_tuple in trial["licks"]])

    # for x in sorted(event_set):
    #     print x

    #     expand_dict(trials, annotate_rewards(trial), index)
    #     expand_dict(trials, annotate_optogenetics(trial), index)
    #     expand_dict(trials, annotate_responses(trial), index)
    #     expand_dict(
    #         trials,
    #         annotate_schedule_time(
    #             trial,
    #             pre_change_time,
    #             initial_blank_duration
    #         ),
    #         index
    #     )
    #     expand_dict(trials, annotate_stimuli(trial, stimuli), index)

    trials = pd.DataFrame(trial_data)
    # print trials

    # trials = trials.rename(
    #     columns={
    #         "start_time": "starttime",
    #         "start_frame": "startframe",
    #         "end_time": "endtime",
    #         "end_frame": "endframe",
    #         "delta_orientation": "delta_ori",
    #         "auto_rewarded_trial": "auto_rewarded",
    #         "change_orientation": "change_ori",
    #         "initial_orientation": "initial_ori",
    #     }
    # ).reset_index()

    return trials