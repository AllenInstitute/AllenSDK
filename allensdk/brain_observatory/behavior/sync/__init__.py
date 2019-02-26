"""
Created on Sunday July 15 2018

@author: marinag
"""

from .process_sync import filter_digital, calculate_delay  # NOQA: E402
from .sync_dataset import Dataset as SyncDataset  # NOQA: E402

def get_sync_data(sync_path, use_acq_trigger=False):
    
    # use_acq_trigger = lims_data.filter_solenoid_2P6

    # sync_path = lims_data.sync_file


    sync_dataset = SyncDataset(sync_path)
    meta_data = sync_dataset.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
    vs2p_f = sync_dataset.get_falling_edges(
        '2p_vsync', )  # new sync may be able to do units = 'sec', so conversion can be skipped
    vs2p_rsec = vs2p_r / sample_freq
    vs2p_fsec = vs2p_f / sample_freq
    if use_acq_trigger:  # if 2P6, filter out solenoid artifacts
        vs2p_r_filtered, vs2p_f_filtered = filter_digital(vs2p_rsec, vs2p_fsec, threshold=0.01)
        frames_2p = vs2p_r_filtered
    else:  # dont need to filter out artifacts in pipeline data
        frames_2p = vs2p_rsec
    # use rising edge for Scientifica, falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
    # Convert to seconds - skip if using units in get_falling_edges, otherwise convert before doing filter digital
    # vs2p_rsec = vs2p_r / sample_freq
    # frames_2p = vs2p_rsec
    # stimulus vsyncs
    # vs_r = d.get_rising_edges('stim_vsync')
    vs_f = sync_dataset.get_falling_edges('stim_vsync')
    # convert to seconds
    # vs_r_sec = vs_r / sample_freq
    vs_f_sec = vs_f / sample_freq
    # vsyncs = vs_f_sec
    # add display lag
    monitor_delay = calculate_delay(sync_dataset, vs_f_sec, sample_freq)
    stimulus_frames = vs_f_sec - monitor_delay # this should be subtracted; makes master timeline pre-delay.
    # line labels are different on 2P6 and production rigs - need options for both
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
        stim_photodiode = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
    elif 'photodiode' in meta_data['line_labels']:
        stim_photodiode = sync_dataset.get_rising_edges('photodiode') / sample_freq
    if 'cam1_exposure' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
    elif 'eye_tracking' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
    if 'cam2_exposure' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
    elif 'behavior_monitoring' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq
    # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger for 2P6 only
    if use_acq_trigger:
        frames_2p = frames_2p[frames_2p > trigger[0]]

    print 'sync_licks:', len(lick_times)

    sync_data = {'ophys_frames': frames_2p,
                 'stimulus_frames': stimulus_frames,
                 'lick_times': lick_times,
                 'ophys_trigger': trigger,
                 'eye_tracking': eye_tracking,
                 'behavior_monitoring': behavior_monitoring,
                 'stim_photodiode': stim_photodiode,
                 }

    return sync_data
