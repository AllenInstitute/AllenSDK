"""
Created on Sunday July 15 2018

@author: marinag
"""

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
    if 'cam1_exposure' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
    elif 'eye_tracking' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
    if 'cam2_exposure' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
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

def get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay):
    
    # Time rebasing: times in stimulus_timestamps_pickle and lick log will agree with times in event log
    vsyncs = data["items"]["behavior"]['intervalsms']
    stimulus_timestamps_pickle_pre = np.hstack((0, vsyncs)).cumsum() / 1000.0

    assert len(stimulus_timestamps_pickle_pre) == len(stimulus_timestamps_no_monitor_delay)
    first_trial = data["items"]["behavior"]["trial_log"][0]
    first_trial_start_time, first_trial_start_frame = {(e[0], e[1]):(e[2], e[3]) for e in first_trial['events']}['trial_start','']
    offset_time = first_trial_start_time-stimulus_timestamps_pickle_pre[first_trial_start_frame]
    stimulus_timestamps_pickle = np.array([t+offset_time for t in stimulus_timestamps_pickle_pre])

    # Rebase used to transform trial log times to sync times:
    time_slope, time_intercept, _, _, _ = sps.linregress(stimulus_timestamps_pickle, stimulus_timestamps_no_monitor_delay)
    def rebase(t):
        return time_intercept+time_slope*t

    return rebase