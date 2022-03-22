# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 13:08:37 2018
@author: svc_ccg
"""

from __future__ import division
from matplotlib import pyplot as plt

from allensdk.brain_observatory.ecephys.data_objects.eye_tracking.ecephys import get_probe_time_offset, extract_barcodes_from_times
import pandas as pd
import numpy as np
import glob, os, json, re
import logging
from xml.dom.minidom import parse
# import visual_behavior

#for vsync alignment
from typing import Union
import numpy as np
import scipy.spatial.distance as distance

def getUnitData(probeBase,syncDataset):

    probeSpikeDir = os.path.join(probeBase, r'continuous\\Neuropix-PXI-100.0')
    
    #Get barcodes/times from probe events and sync file
    be_t, be = get_ephys_barcodes(probeBase)
    be_t, be = cut_bad_barcodes(be_t, be, 'ephys', threshold=30.8)
    
    bs_t, bs = get_sync_barcodes(syncDataset)
    bs_t, bs = cut_bad_barcodes(bs_t, bs, 'sync')
    
    
    #Compute time shift between ephys and sync
    shift, p_sampleRate, m_endpoints = get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
    
    #Get unit spike times 
    units = load_spike_info(probeSpikeDir, p_sampleRate, shift)
    
    return units


def get_ephys_barcodes(probeBase):

    probeTTLDir = os.path.join(probeBase, r'events\\Neuropix-PXI-100.0\\TTL_1')
    
    channel_states = np.load(os.path.join(probeTTLDir, 'channel_states.npy'))
    event_times = np.load(os.path.join(probeTTLDir, 'event_timestamps.npy'))
    
    beRising = event_times[channel_states>0]/30000.
    beFalling = event_times[channel_states<0]/30000.
    be_t, be = extract_barcodes_from_times(beRising, beFalling)

    return be_t, be


def get_sync_barcodes(sync_dataset, fallback_line=0):
    
    lines = sync_dataset.line_labels
    
    #look for barcodes in labels
    bline = fallback_line
    for line in lines:
        if 'barcode' in line:
            bline = line
    
    bRising = sync_dataset.get_rising_edges(bline, units='seconds')
    bFalling = sync_dataset.get_falling_edges(bline, units='seconds')
    
    bs_t, bs = extract_barcodes_from_times(bRising, bFalling)
    
    return bs_t, bs


def cut_bad_barcodes(bs_t, bs, source, threshold=30.95):
    
    if any(np.diff(bs_t)<30.95):
        logging.warning('Detected bad barcode interval in {}, truncating data'.format(source))
        
        #find bad barcodes
        bad_intervals = np.where(np.diff(bs_t)<threshold)[0]
        bad_barcode_indices = [bi+1 for bi in bad_intervals]
        
        #find largest block of good barcodes to use for probe sample rate/offset
        bbi = np.insert(bad_barcode_indices, 0, 0)
        bbi = np.append(bbi, len(bs_t))
        good_block_sizes = np.diff(bbi)
        largest_block = np.argmax(good_block_sizes)
        barcode_interval_to_use = [bbi[largest_block], bbi[largest_block+1]-1]
        
        bs_t = bs_t[barcode_interval_to_use[0]:barcode_interval_to_use[1]]
        bs = bs[barcode_interval_to_use[0]:barcode_interval_to_use[1]]
    
    return bs_t, bs


def build_unit_table(probes_to_run, paths, syncDataset):
    ### GET UNIT METRICS AND BUILD UNIT TABLE ###
    probe_dirs = [[paths['probe'+pid], pid] for pid in probes_to_run]
    probe_dict = {a[1]:{} for a in probe_dirs}
    successful_probes = []
    for p in probe_dirs:
        print(p)
        try:
            print('########## Getting Units for probe {} ###########'.format(p[1]))
            probe = p[1]
            full_path = p[0]
            
            # Get unit metrics for this probe    
            metrics_file = os.path.join(full_path, 'continuous\\Neuropix-PXI-100.0\\metrics.csv')
            unit_metrics = pd.read_csv(metrics_file)
            unit_metrics = unit_metrics.set_index('cluster_id')
            
            # Get unit data
            units = getUnitData(full_path, syncDataset)
            units = pd.DataFrame.from_dict(units, orient='index')
            units['cluster_id'] = units.index.astype(int)
            units = units.set_index('cluster_id')
#            units['probe'] = p
#            units['uid'] = units['probe'] + units.index.astype(str)
            
            units = pd.merge(unit_metrics, units, left_index=True, right_index=True, how='outer')
            units['probe'] = probe
            units['uid'] = units['probe'] + units.index.astype(str)
            units = units.set_index('uid')
            
            probe_dict[probe] = units
            successful_probes.append(probe)
        except Exception as E:
            logging.error(E)
    
        
        
    #return  {k:probe_dict[k] for k in successful_probes}
    return pd.concat([probe_dict[k] for k in successful_probes])
            

def map_probe_from_slot_port(pinfo):
    
    probenames = [None, None,' ABC',' DEF']
    slot = int(pinfo['slot'])
    port = int(pinfo['port'])
    
    probename = probenames[slot][port]
    return probename
    

def get_probe_settings_from_xml(xmlfilepath):
    
    settings = parse(xmlfilepath)
    probes = settings.getElementsByTagName('PROBE')
    probe_info_dict = {}
    for probe in probes:
        pinfo = {}
        for attr in probe.attributes.items(): 
            pinfo[attr[0]] = attr[1]
            
        probename = map_probe_from_slot_port(pinfo)
        probe_info_dict[probename] = pinfo
    
    return probe_info_dict

    
def get_sync_line_data(syncDataset, line_label=None, channel=None):
    ''' Get rising and falling edge times for a particular line from the sync h5 file
        
        Parameters
        ----------
        dataset: sync file dataset generated by sync.Dataset
        line_label: string specifying which line to read, if that line was labelled during acquisition
        channel: integer specifying which channel to read in line wasn't labelled
        
        Returns
        ----------
        rising: npy array with rising edge times for specified line
        falling: falling edge times
    '''
    if isinstance(line_label, str):
        try:
            channel = syncDataset.line_labels.index(line_label)
        except:
            print('Invalid line label')
            return
    elif channel is None:
        print('Must specify either line label or channel id')
        return
    
    rising = syncDataset.get_rising_edges(channel, units='seconds')
    falling = syncDataset.get_falling_edges(channel, units='seconds')
    
    return rising, falling


def load_spike_info(spike_data_dir, p_sampleRate, shift):
    ''' Make dictionary with spike times, templates, sorting label and peak channel for all units
    
        Parameters
        -----------
        spike_data_dir: path to directory with clustering output files
        p_sampleRate: probe sampling rate according to master clock
        shift: time shift between master and probe clock
        p_sampleRate and shift are outputs from 'get_probe_time_offset' function
        sortMode: if KS, read in automatically generated labels from Kilosort; if phy read in phy labels
        
        Returns
        ----------
        units: dictionary with spike info for all units
            each unit is integer key, so units[0] is a dictionary for spike cluster 0 with keys
            'label': sorting label for unit, eg 'good', 'mua', or 'noise'
            'times': spike times in seconds according to master clock
            'template': spike template, should be replaced by waveform extracted from raw data
                averaged over 1000 randomly chosen spikes
            'peakChan': channel where spike template has minimum, used to approximate unit location
    '''
    print(p_sampleRate)
    print(shift)
    spike_clusters = np.load(os.path.join(spike_data_dir, 'spike_clusters.npy'))
    spike_times = np.load(os.path.join(spike_data_dir, 'spike_times.npy'))
    templates = np.load(os.path.join(spike_data_dir, 'templates.npy'))
    spike_templates = np.load(os.path.join(spike_data_dir, 'spike_templates.npy'))
    channel_positions = np.load(os.path.join(spike_data_dir, 'channel_positions.npy'))
    amplitudes = np.load(os.path.join(spike_data_dir, 'amplitudes.npy'))
    unit_ids = np.unique(spike_clusters)
    
    units = {}
    for u in unit_ids:
        ukey = str(u)
        units[ukey] = {}
    
        unit_idx = np.where(spike_clusters==u)[0]
        unit_sp_times = spike_times[unit_idx]/p_sampleRate - shift
        
        units[ukey]['times'] = unit_sp_times
        
        #choose 1000 spikes with replacement, then average their templates together
        chosen_spikes = np.random.choice(unit_idx, 1000)
        chosen_templates = spike_templates[chosen_spikes].flatten()
        units[ukey]['template'] = np.mean(templates[chosen_templates], axis=0)
        units[ukey]['peakChan'] = np.unravel_index(np.argmin(units[ukey]['template']), units[ukey]['template'].shape)[1]
        units[ukey]['position'] = channel_positions[units[ukey]['peakChan']]
        units[ukey]['amplitudes'] = amplitudes[unit_idx]
        
#        #check if this unit is noise
#        peakChan = units[ukey]['peakChan']
#        temp = units[ukey]['template'][:, peakChan]
#        pt = findPeakToTrough(temp, plot=False)
#        units[ukey]['peakToTrough'] = pt

        
    return units
    
    
def getLFPData(probeBase, syncDataset, num_channels=384):
    
    probeTTLDir = os.path.join(probeBase, r'events\\Neuropix-PXI-100.0\\TTL_1')
    lfp_data_dir = os.path.join(probeBase, r'continuous\\Neuropix-PXI-100.1')
    lfp_data_file = os.path.join(lfp_data_dir, 'continuous.dat')
    
    
    if not os.path.exists(lfp_data_file):
        print('Could not find LFP data at ' + lfp_data_file)
        return None,None
    
    lfp_data = np.memmap(lfp_data_file, dtype='int16', mode='r')    
    lfp_data_reshape = np.reshape(lfp_data, [int(lfp_data.size/num_channels), -1])
    time_stamps = np.load(os.path.join(lfp_data_dir, 'lfp_timestamps.npy'))
        
    
    bRising, bFalling = get_sync_line_data(syncDataset, channel=0)
    bs_t, bs = extract_barcodes_from_times(bRising, bFalling)
    
    channel_states = np.load(os.path.join(probeTTLDir, 'channel_states.npy'))
    event_times = np.load(os.path.join(probeTTLDir, 'event_timestamps.npy'))
    
    beRising = event_times[channel_states>0]/30000.
    beFalling = event_times[channel_states<0]/30000.
    be_t, be = extract_barcodes_from_times(beRising, beFalling)
    
    
    #Compute time shift between ephys and sync
    shift, p_sampleRate, m_endpoints = get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
    
    
    time_stamps_shifted = (time_stamps/p_sampleRate) - shift
    
    return lfp_data_reshape, time_stamps_shifted


def build_lfp_dict(probe_dirs, syncDataset):
    
    lfp_dict = {}

    for ip, probe in enumerate(probe_dirs):
        #p_name = probe.split('_')[-2][-1]
        p_name = re.findall('probe[A-F]', probe)[0][-1]
        
        lfp, time = getLFPData(probe, syncDataset)
        lfp_dict[p_name] = {'time': time, 'lfp': lfp}
    
    return lfp_dict


def get_surface_channels(probe_dirs):
    pass


def get_frame_offsets(sync_dataset, frame_counts, tolerance=0):
    ''' Tries to infer which vsyncs correspond to the frames in the epochs in frame_counts
        This allows you to align data even when there are aborted stimuli
        
        INPUTS:
            sync_dataset: sync data from experiment (a 'Dataset' object made from the H5 file)
            
            frame_counts: list of the expected frame counts (taken from pkl files) for each
                        of the stimuli in question;
                        the list should be ordered by the display sequence
            
            tolerance: percent by which frame counts are allowed to deviate from expected
                        
        OUTPUTS:
            start_frames: list of the inferred start frames for each of the stimuli
    '''
    
    frame_counts = np.array(frame_counts)
    tolerance = tolerance/100.
    
    # get vsyncs and stim_running signals from sync
    vf = get_vsyncs(sync_dataset)
    stimstarts, stimoffs = get_stim_starts_ends(sync_dataset)
    print(stimstarts)
    print(stimoffs)
    print(len(vf))
    
    # get vsync frame lengths for all stimuli
    epoch_frame_counts = []
    epoch_start_frames = []
    for start, end in zip(stimstarts, stimoffs):
        epoch_frames = np.where((vf>start)&(vf<end))[0]
        epoch_frame_counts.append(len(epoch_frames))
        epoch_start_frames.append(epoch_frames[0])
    print(epoch_frame_counts)
    print(frame_counts)
        
    if len(epoch_frame_counts)>len(frame_counts):
        logging.warning('Found extra stim presentations. Inferring start frames')
        
        start_frames = []
        for stim_num, fc in enumerate(frame_counts):
            
            print('finding stim start for stim {}'.format(stim_num))
            best_match = np.argmin([np.abs(e-fc) for e in epoch_frame_counts])
            if fc*(1-tolerance) <= epoch_frame_counts[best_match] <= fc*(1+tolerance):
                _ = epoch_frame_counts.pop(best_match)
                start_frame = epoch_start_frames.pop(best_match)
                start_frames.append(start_frame)
                print('found stim start at vsync {}'.format(start_frame))
                
            else:
                logging.error('Could not find matching sync frames for stim {}'.format(stim_num))
                return
    
    
    else:        
        start_frames = epoch_start_frames
    
    return start_frames


def get_bad_vsync_indices(sync_dataset):
    '''find bad vsyncs if the sync drops data'''
    
    bs_t, bs = get_sync_barcodes(sync_dataset)
    barcode_intervals = np.diff(bs_t)
    median_barcode_interval = np.median(barcode_intervals)
    bad_intervals = np.where(barcode_intervals<30.95)[0]
    
    bad_barcode_indices = [bi+1 for bi in bad_intervals]
    
    
    bad_barcode_intervals = []
    for bi in bad_barcode_indices:
        
        #find last good barcode before bad one
        last_good_barcode = bi
        while last_good_barcode in bad_barcode_indices:
            last_good_barcode = last_good_barcode - 1
            
        #find next good barcode after bad one
        next_good_barcode = bi
        while next_good_barcode in bad_barcode_indices:
            next_good_barcode = next_good_barcode + 1
        
        bad_barcode_intervals.append([last_good_barcode, next_good_barcode])
    
    
    #find the indices for the vsyncs that need to be interpolated
    bad_synctime_intervals = [[bs_t[a], bs_t[b]] for a,b in bad_barcode_intervals]
    time_lost_per_interval = [(b-a)*median_barcode_interval for a,b in bad_barcode_intervals]
    vsyncs = get_vsyncs(sync_dataset)
    vsync_patch_indices = [[np.searchsorted(vsyncs, a), np.searchsorted(vsyncs, b)] for a,b in bad_synctime_intervals]
    
    return vsync_patch_indices, time_lost_per_interval


def patch_vsyncs(sync_dataset, behavior_data, mapping_data, replay_data):
    
    '''Hack to patch bad vsync intervals if sync drops data'''
    
    behavior_vsync_intervals = behavior_data['items']['behavior']['intervalsms']
    mapping_vsync_intervals = mapping_data['intervalsms']
    replay_vsync_intervals = replay_data['intervalsms']
    
    concatenated_intervals = np.concatenate((behavior_vsync_intervals, [np.nan], 
                                             mapping_vsync_intervals, [np.nan],
                                             replay_vsync_intervals))/1000.
    vsyncs = get_vsyncs(sync_dataset)
    vsync_intervals = np.diff(vsyncs)
    
    bad_vsync_indices, time_lost_per_interval = get_bad_vsync_indices(sync_dataset)
    for bad_inds, time_lost in zip(bad_vsync_indices, time_lost_per_interval):
        
        bad_ind_start, bad_ind_end = bad_inds
        
        #cut out bad section
        vsync_intervals = np.concatenate((vsync_intervals[:bad_ind_start], vsync_intervals[bad_ind_end:]))
        
        #paste in vsyncs from the pickle files
        pkl_start_ind = bad_ind_start
        pkl_end_ind = bad_ind_start
        pkl_time = 0
        while pkl_time<time_lost:
            pkl_end_ind = pkl_end_ind + 1
            pkl_time = np.sum(concatenated_intervals[pkl_start_ind:pkl_end_ind])
        
        vsync_intervals = np.insert(vsync_intervals, bad_ind_start, 
                           concatenated_intervals[pkl_start_ind:pkl_end_ind])
    
    vsyncs_corrected = vsyncs[0] + np.cumsum(np.insert(vsync_intervals, 0, 0))
    return vsyncs_corrected
        

def get_running_from_pkl(pkl):
    key = 'behavior' if 'behavior' in pkl['items'] else 'foraging'
    intervals = pkl['items']['behavior']['intervalsms'] if 'intervalsms' not in pkl else pkl['intervalsms']
    time = np.insert(np.cumsum(intervals), 0, 0)/1000.
    
    dx,vsig,vin = [pkl['items'][key]['encoders'][0][rkey] for rkey in ('dx','vsig','vin')]
    run_speed = visual_behavior.analyze.compute_running_speed(dx[:len(time)],time,vsig[:len(time)],vin[:len(time)])
    
    return dx, run_speed
    

def get_vsyncs(sync_dataset, fallback_line=2):
    
    lines = sync_dataset.line_labels
    
    #look for vsyncs in labels
    vsync_line = fallback_line
    for line in lines:
        if 'vsync' in line:
            vsync_line = line
    
    rising_edges = sync_dataset.get_rising_edges(vsync_line, units='seconds')
    falling_edges = sync_dataset.get_falling_edges(vsync_line, units='seconds')
    
    #ignore the first falling edge if it isn't preceded by a rising edge
    return falling_edges[falling_edges>rising_edges[0]]
        

def get_stim_starts_ends(sync_dataset, fallback_line=5):
    
    lines = sync_dataset.line_labels
    
    #look for vsyncs in labels
    if 'stim_running' in lines:
        stim_line = 'stim_running'
    else:
        stim_line = fallback_line
    
    stim_ons = sync_dataset.get_rising_edges(stim_line, units='seconds')
    stim_offs = sync_dataset.get_falling_edges(stim_line, units='seconds')

    if stim_offs[0]<stim_ons[0]:
        logging.warning('Found extra stim off. Truncating.')
        stim_offs = stim_offs[1:]

    if len(stim_offs) != len(stim_ons):
        logging.warning('Found {} stim starts, but {} stim offs. \
            Sync signal is suspect...'.format(len(stim_ons), len(stim_offs)))

    return stim_ons, stim_offs


def get_diode_times(sync_dataset, fallback_line=4):
    
    lines = sync_dataset.line_labels
    
    diode_line = fallback_line
    for line in lines:
        if 'photodiode' in line:
            diode_line = line
    
    rising_edges = sync_dataset.get_rising_edges(diode_line, units='seconds')
    falling_edges = sync_dataset.get_falling_edges(diode_line, units='seconds')
    
    return rising_edges, falling_edges


def get_monitor_lag(syncDataset):

    dioder, diodef = get_diode_times(syncDataset)
    vf = get_vsyncs(syncDataset)
    
    lag = np.min([np.min(np.abs(d-vf[60])) for d in [diodef, dioder]])
    
    return lag


def get_lick_times(sync_dataset, fallback_line=31):
    
    lines = sync_dataset.line_labels
    
    lick_line = fallback_line
    for line in lines:
        if 'lick' in line:
            lick_line = line
    
    lick_times = sync_dataset.get_rising_edges(lick_line, units='seconds')
    
    return lick_times


### FUNCTIONS TO GET THE FRAME TIMES AND REMOVE DROPPED FRAMES
def extract_lost_frames_from_json(cam_json):
    
    lost_count = cam_json['RecordingReport']['FramesLostCount']
    if lost_count == 0:
        return []
    
    lost_string = cam_json['RecordingReport']['LostFrames'][0]
    lost_spans = lost_string.split(',')
    
    lost_frames = []
    for span in lost_spans:
        
        start_end = span.split('-')
        if len(start_end)==1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1])+1))
    
    return np.array(lost_frames)-1 #you have to subtract one since the json starts indexing at 1 according to Totte
    

def get_frame_exposure_times(sync_dataset, cam_json):
    
    if isinstance(cam_json, str):
        cam_json = read_json(cam_json)
        
    exposure_sync_line_label_dict = {
            'Eye': 'eye_cam_exposing',
            'Face': 'face_cam_exposing',
            'Behavior': 'beh_cam_exposing'}
    
    cam_label =  cam_json['RecordingReport']['CameraLabel']
    sync_line = exposure_sync_line_label_dict[cam_label]
    
    exposure_times = sync_dataset.get_rising_edges(sync_line, units='seconds')
    
    lost_frames = extract_lost_frames_from_json(cam_json)
    
    frame_times = [e for ie, e in enumerate(exposure_times) if ie not in lost_frames]
    
    return np.array(frame_times)


def read_json(jsonfilepath):
    
    with open(jsonfilepath, 'r') as f:
        contents = json.load(f)
    
    return contents
    

###Functions to improve frame syncing###
def get_experiment_frame_times(sync, photodiode_cycle=60):
    
    photodiode_times = np.sort(np.concatenate([
                    sync.get_rising_edges('stim_photodiode', 'seconds'),
                    sync.get_falling_edges('stim_photodiode', 'seconds')
                ]))
    vsync_times = sync.get_falling_edges('vsync_stim', 'seconds')
    stimstarts, stimends = get_stim_starts_ends(sync)
    ccb_frame_times = []
    
    for ie, (start, end) in enumerate(zip(stimstarts, stimends)):
        
        epoch_vsyncs = vsync_times[(vsync_times>=start)&(vsync_times<=end)]
        epoch_photodiodes = photodiode_times[(photodiode_times>=start)&(photodiode_times<=end)]
        
        frame_duration = estimate_frame_duration(
                epoch_photodiodes, cycle=photodiode_cycle)
        
        ccb_times = get_ccb_frame_times(epoch_vsyncs, epoch_photodiodes, photodiode_cycle, frame_duration)
        
        ccb_frame_times.append(ccb_times)
    
    all_ccb_times = np.concatenate(ccb_frame_times)
    
    return all_ccb_times
    
    
 
def get_ccb_frame_times(vsyncs, photodiode_times, photodiode_cycle, frame_duration):
    
    #removes blinking at beginning and end of each stimulus
    photodiode_times = trim_border_pulses(
                photodiode_times, vsyncs
            )
    
    # not totally sure... correcting for on/off photodiode asymmetry
    photodiode_times = correct_on_off_effects(
        photodiode_times
    )
    
    # fix blips in the line
    photodiode_times = fix_unexpected_edges(
        photodiode_times, cycle=photodiode_cycle)
    
    
    return compute_frame_times_ccb(vsyncs, photodiode_times, frame_duration, 60)


def compute_frame_times_ccb(
    vsyncs, photodiode_times, frame_duration, cycle,
):
    
    num_frames = len(vsyncs)
    starts = np.zeros(num_frames, dtype=float)
    vdiffs = np.diff(vsyncs)
    
    #trim photodiode times to make sure there are no strays at the end
    photodiode_times = photodiode_times[:int(np.floor(len(vsyncs)/60)+1)]
    print('num photodiode intervals used {}'.format(len(photodiode_times)))
    
    for start_index, (start_time, end_time) in enumerate(
        zip(photodiode_times[:-1], photodiode_times[1:])
    ):

        interval_duration = end_time - start_time
        these_vsyncs = vsyncs[start_index * cycle: (start_index + 1) * cycle]
        these_vdiffs = vdiffs[start_index * cycle: (start_index + 1) * cycle]
       # frame_duration = np.median(np.diff(these_vsyncs))
        long_frame_time = np.sum(these_vdiffs) - cycle*frame_duration 
        long_frame_time_in_frames = int(np.round(long_frame_time/frame_duration))
        
        extra_time_in_frames =(
            int(np.around((interval_duration) / frame_duration))
            - cycle
        )
        
        extra_monitor_lag_in_frames = 0
        if long_frame_time_in_frames<extra_time_in_frames:
            print('extra monitor lag detected')
            print(long_frame_time_in_frames)
            print(extra_time_in_frames)
            print(start_time)
            extra_monitor_lag_in_frames = extra_time_in_frames - long_frame_time_in_frames
            
        
            
        local_frame_duration = interval_duration / (cycle + extra_time_in_frames)
        #ideal_vsyncs = np.arange(start_time, end_time, frame_duration)
        if extra_time_in_frames > 0:
            
            #find long frames and shift times accordingly
            frame_diffs = np.round(np.diff(these_vsyncs)/local_frame_duration)
            relative_vsyncs = np.insert(np.cumsum(frame_diffs*local_frame_duration), 0, 0)
            
            #assume that if there was a change in monitor lag, it was after the long frame
            longest_ind = np.argmax(these_vdiffs)+1
            relative_vsyncs[longest_ind:] += extra_monitor_lag_in_frames*local_frame_duration
            
        
        else:
            frame_diffs = np.ones(cycle-1)
            relative_vsyncs = np.insert(np.cumsum(frame_diffs*local_frame_duration), 0, 0)
        
        starts[start_index * cycle: (start_index + 1) * cycle] = relative_vsyncs + start_time
    
    
    #Now deal with leftover frames that occur after the last diode transition
    #Just take the global frame duration for these
    leftover_frames_first_ind = len(starts) - np.mod(len(starts), cycle)
    these_vsyncs = vsyncs[leftover_frames_first_ind:]
    frame_diffs = np.round(np.diff(these_vsyncs)/frame_duration)
    print('processing {} leftover frames after last diode transition'.format(len(these_vsyncs)))
    relative_vsyncs = np.insert(np.cumsum(frame_diffs*frame_duration), 0, 0)
    starts[leftover_frames_first_ind:] = photodiode_times[-1]+relative_vsyncs
    
    return starts


def trim_border_pulses(pd_times, vs_times, frame_interval=1/60, num_frames=3):
    pd_times = np.array(pd_times)
    pd_times = pd_times[np.logical_and(
        pd_times >= vs_times[0],
        pd_times <= vs_times[-1] + num_frames * frame_interval
    )]
    
    print('last photodiode transition {}'.format(pd_times[-1]))
    print('last vsyncs time plus buffer {}'.format(vs_times[-1] + num_frames * frame_interval))
    print('num expected diode transitions {}'.format(np.floor(len(vs_times)/60)+1))
    print('num actual diode transitions {}'.format(len(pd_times)))
    return pd_times


def correct_on_off_effects(pd_times):
    """
    Notes
    -----
    This cannot (without additional info) determine whether an assymmetric
    offset is odd-long or even-long.
    """

    pd_diff = np.diff(pd_times)
    odd_diff_mean, odd_diff_std = trimmed_stats(pd_diff[1::2])
    even_diff_mean, even_diff_std = trimmed_stats(pd_diff[0::2])

    half_diff = np.diff(pd_times[0::2])
    full_period_mean, full_period_std = trimmed_stats(half_diff)
    half_period_mean = full_period_mean / 2

    odd_offset = odd_diff_mean - half_period_mean
    even_offset = even_diff_mean - half_period_mean

    pd_times[::2] -= odd_offset / 2
    pd_times[1::2] -= even_offset / 2

    return pd_times


def flag_unexpected_edges(pd_times, ndevs=10):
    pd_diff = np.diff(pd_times)
    diff_mean, diff_std = trimmed_stats(pd_diff)

    expected_duration_mask = np.ones(pd_diff.size)
    expected_duration_mask[np.logical_or(
        pd_diff < diff_mean - ndevs * diff_std,
        pd_diff > diff_mean + ndevs * diff_std
    )] = 0
    expected_duration_mask[1:] = np.logical_and(
        expected_duration_mask[:-1], expected_duration_mask[1:]
    )
    expected_duration_mask = np.concatenate(
        [expected_duration_mask, [expected_duration_mask[-1]]]
    )

    return expected_duration_mask


def fix_unexpected_edges(pd_times, ndevs=10, cycle=60, max_frame_offset=4):
    pd_times = np.array(pd_times)
    expected_duration_mask = flag_unexpected_edges(pd_times, ndevs=ndevs)
    diff_mean, diff_std = trimmed_stats(np.diff(pd_times))
    frame_interval = diff_mean / cycle

    bad_edges = np.where(expected_duration_mask == 0)[0]
    bad_blocks = np.sort(np.unique(np.concatenate([
        [0],
        np.where(np.diff(bad_edges) > 1)[0] + 1,
        [len(bad_edges)]
    ])))

    output_edges = []
    for low, high in zip(bad_blocks[:-1], bad_blocks[1:]):
        current_bad_edge_indices = bad_edges[low: high-1]
        current_bad_edges = pd_times[current_bad_edge_indices]
        low_bound = pd_times[current_bad_edge_indices[0]]
        high_bound = pd_times[current_bad_edge_indices[-1] + 1]

        edges_missing = int(np.around((high_bound - low_bound) / diff_mean))
        expected = np.linspace(low_bound, high_bound, edges_missing + 1)

        distances = distance.cdist(
            current_bad_edges[:, None], expected[:, None]
        )
        distances = np.around(distances / frame_interval).astype(int)

        min_offsets = np.amin(distances, axis=0)
        min_offset_indices = np.argmin(distances, axis=0)
        output_edges = np.concatenate([
            output_edges,
            expected[min_offsets > max_frame_offset],
            current_bad_edges[
                min_offset_indices[min_offsets <= max_frame_offset]
            ]
        ])

    return np.sort(
        np.concatenate([output_edges, pd_times[expected_duration_mask > 0]])
    )

def trimmed_stats(data, pctiles=(10, 90)):
    low = np.percentile(data, pctiles[0])
    high = np.percentile(data, pctiles[1])

    trimmed = data[np.logical_and(
        data <= high,
        data >= low
    )]

    return np.mean(trimmed), np.std(trimmed)   

def estimate_frame_duration(pd_times, cycle=60):
    return trimmed_stats(np.diff(pd_times))[0] / cycle