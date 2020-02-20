"""
Created on Sunday July 15 2018

@author: marinag
"""
from itertools import chain
from typing import Dict, Any
from .process_sync import filter_digital, calculate_delay  # NOQA: E402
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset  # NOQA: E402
import numpy as np
import scipy.stats as sps

def get_sync_data(sync_path):
    
    sync_dataset = SyncDataset(sync_path)
    meta_data = sync_dataset.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    
    # use rising edge for Scientifica, falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
    # 2P vsyncs
    vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
    vs2p_f = sync_dataset.get_falling_edges('2p_vsync')  # new sync may be able to do units = 'sec', so conversion can be skipped
    frames_2p = vs2p_r / sample_freq
    vs2p_fsec = vs2p_f / sample_freq

    stimulus_times_no_monitor_delay = sync_dataset.get_falling_edges('stim_vsync') / sample_freq

    if 'lick_times' in meta_data['line_labels']:
        lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
    elif 'lick_sensor' in meta_data['line_labels']:
        lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
    else:
        lick_times = None
    if '2p_trigger' in meta_data['line_labels']:
        trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
    elif 'acq_trigger' in meta_data['line_labels']:
        trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
    if 'stim_photodiode' in meta_data['line_labels']:
        a = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
        b = sync_dataset.get_falling_edges('stim_photodiode') / sample_freq
        stim_photodiode = sorted(list(a)+list(b))
    elif 'photodiode' in meta_data['line_labels']:
        a = sync_dataset.get_rising_edges('photodiode') / sample_freq
        b = sync_dataset.get_falling_edges('photodiode') / sample_freq
        stim_photodiode = sorted(list(a)+list(b))
    if 'cam2_exposure' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
    elif 'eye_tracking' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
    if 'cam1_exposure' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
    elif 'behavior_monitoring' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq

    sync_data = {'ophys_frames': frames_2p,
                 'lick_times': lick_times,
                 'ophys_trigger': trigger,
                 'eye_tracking': eye_tracking,
                 'behavior_monitoring': behavior_monitoring,
                 'stim_photodiode': stim_photodiode,
                 'stimulus_times_no_delay': stimulus_times_no_monitor_delay,
                 }

    return sync_data


def frame_time_offset(data: Dict[str, Any]) -> float:
    """
    Contained in the behavior "pickle" file is a series of time between
    consecutive vsync frames (`intervalsms`). This information required
    to get the timestamp (via frame number) for events that occured
    outside of a trial(e.g. licks). However, we don't have the value
    in the trial log time stream when the first vsync frame actually
    occured -- so we estimate it with a linear regression (frame
    number x time). All trials in the `trial_log` have events for
    `trial_start` and `trial_end`, so these are used to fit the
    regression. A linear regression is used rather than just
    subtracting the time from the first trial, since there can be some
    jitter given the 60Hz refresh rate.

    Parameters
    ----------
    data: dict
        behavior pickle well-known file data

    Returns
    -------
    float
        Time offset to add to the vsync stream to sync it with the
        `trial_log` time stream. The "zero-th" frame time.
    """
    events = [trial["events"] for trial
              in data["items"]["behavior"]["trial_log"]]
    # First event in `events` is `trial_start`, and last is `trial_end`
    # Event log has following schema:
    #    event name, description: 'enter'/'exit'/'', time, frame number
    # We want last two fields for first and last trial event
    trial_by_frame = list(chain(
        [event[i][-2:] for event in events for i in [0, -1]]))
    times = [trials[0] for trials in trial_by_frame]
    frames = [trials[1] for trials in trial_by_frame]
    time_to_first_vsync = sps.linregress(frames, times).intercept
    return time_to_first_vsync


def get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay):
    """
    Create a rebase function to align times for licks and stimulus timestamps
    in the "pickle" log with the same events in the event "sync" log.
    """
    vsyncs = data["items"]["behavior"]['intervalsms']
    stimulus_timestamps_pickle_pre = np.hstack((0, vsyncs)).cumsum() / 1000.0

    if (len(stimulus_timestamps_pickle_pre)
            != len(stimulus_timestamps_no_monitor_delay)):
        raise ValueError("Number of stimulus timestamps in pickle file are "
                         "not equal to timestamps in the sync file (pickle="
                         f"{len(stimulus_timestamps_pickle_pre)}, sync="
                         f"{len(stimulus_timestamps_no_monitor_delay)}).")
    time_to_first_vsync = frame_time_offset(data)
    stimulus_timestamps_pickle = (stimulus_timestamps_pickle_pre
                                  + time_to_first_vsync)
    # Rebase used to transform trial log times to sync times:
    time_slope, time_intercept, _, _, _ = sps.linregress(
        stimulus_timestamps_pickle, stimulus_timestamps_no_monitor_delay)

    # return transform of trial log time to sync time
    return lambda t: time_intercept + time_slope * t
