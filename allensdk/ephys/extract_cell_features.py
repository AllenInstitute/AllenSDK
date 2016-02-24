#!/usr/bin/env python

# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import h5py
import sys
import json
import numpy as np
from scipy.optimize import curve_fit
from collections import Counter
import feature_extractor as fx
import logging

from allensdk.core.nwb_data_set import NwbDataSet


def extract_cell_features(nwb_file, long_square_sweeps, short_square_sweeps, ramp_sweeps):
    '''
    Extract features of long square, short square, and ramp sweeps used for cell-wide characterization.
    
    Parameters
    ----------

    nwb_file: string
        File name of NWB file.

    long_square_sweeps: list
        List of sweep numbers of long square sweeps in an NWB file.
        
    short_square_sweeps: list
        List of sweep numbers of short square sweeps in an NWB file.

    ramp_sweeps: list
        List of sweep numbers of ramp sweeps in an NWB file.

    Returns
    -------

    dict: dictionary of properties grouped by sweep type

    
    '''

    long_square_features = analyze_long_squares(long_square_sweeps, nwb_file)

    if "hero_sweep_num" not in long_square_features:
        logging.error("could not identify a hero sweep")
        return None

    if "input_resistance" not in long_square_features:
        logging.error("could not compute input resistance")
        return None

    short_square_features = analyze_short_squares(short_square_sweeps, nwb_file)

    if short_square_features.get("up_down_ratio",None) is None:
        logging.error("could not compute short square up/down ratio: most likely short square stimuli did not trigger a spike")
        return None

    ramp_features = analyze_ramps(ramp_sweeps, nwb_file)

    return {
        'long_squares': long_square_features,
        'ramps': ramp_features,
        'short_squares': short_square_features
        }


def extract_sweep_features(nwb_file, sweep_numbers):
    '''
    Run feature extraction on a per sweep basis for a set of sweeps.  This does not
    feature aggregation for the set of the sweeps.

    Parameters
    ----------

    nwb_file: string
        File name of an NWB file

    sweep_numbers: list
        List of sweeps numbers.
    
    '''

    features = fx.EphysFeatureExtractor()
    all_sweep_features = {}

    for sweep_number in sweep_numbers:
        logging.debug("extracting features for sweep %d" % sweep_number)
        sweep_features = extract_single_sweep_features(features, nwb_file, sweep_number)
        all_sweep_features[sweep_number] = sweep_features

    return all_sweep_features

def extract_single_sweep_features(features, nwb_file, sweep_number):
    '''
    Run feature extraction on a single sweep.  

    Parameters
    ----------
    
    features: EphysFeatureExtractor instance
    
    nwb_file: string
        File name of an NWB file

    sweep_numbers: int
        Sweep number in the NWB file
       
    '''

    nwb = NwbDataSet(nwb_file)
    data = nwb.get_sweep(sweep_number)

    v = data['response']
    curr = data['stimulus']

    idx0 = data['index_range'][0]
    idx1 = data['index_range'][1]

    if idx0 >= idx1:
        logging.warning("Sweep %s stop index precedes start index, skipping spike identification" % sweep_number)
        return 

    hz = data['sampling_rate']
    dt = 1.0 / hz
    t = np.arange(0, len(v)) * dt
    
    features.process_instance(sweep_number, v*1e3, curr*1e12, t, dt*idx0, dt*(idx1-idx0-2), None)
    
    results = {}
    results["mean"] = features.feature_list[-1].mean
    results["stdev"] = features.feature_list[-1].stdev

    return results


def analyze_long_squares(sweep_numbers, nwb_file):
    '''
    Extract features specific to long square sweeps.  From the list of sweeps, this
    will identify the rheobase sweep (lowest-amplitude stimulus inducing a spike), 
    a "hero" sweep (lowest-amplitude sweep 30-60pA above rheobase), and
    a large number of aggregate spike features for the supplied set of sweep numbers.

    Parameters
    ----------

    sweep_numbers: list
        List of long square sweep numbers.

    nwb_file: string
        File name of NWB file.
    '''

    analysis = {}

    if len(sweep_numbers) == 0:
        return analysis
    
    analysis['sweep_info'] = []
    analysis['subthresh'] = []
    analysis['fI'] = []
    features = fx.EphysFeatureExtractor()
    
    # Go through the sweeps to identify the subthreshold ones and the rheobase sweep
    rheo_amp = 1e12
    rheo_n_spikes = 1e12
    rheo_sweep_num = -1

    sweep_numbers = sorted(sweep_numbers)
    for sweep_number in sweep_numbers:

        sweep_info = {}
        sweep_info['sweep_num'] = sweep_number

        v, i, t = get_sweep_from_nwb(nwb_file, sweep_number)

        sweep_info['stim_start'], sweep_info['stim_dur'], sweep_info['stim_amp'], start_idx, end_idx = get_square_stim_characteristics(i, t)

        features.process_instance(sweep_number, v, i, t, 
                                  sweep_info['stim_start'], sweep_info['stim_dur'], 
                                  "long_square")

        n_spikes = features.feature_list[-1].mean['n_spikes']


        steady_interval_start_idx = np.nonzero(t >= sweep_info['stim_start'] + sweep_info['stim_dur'] - 0.1)[0][0]

        # if there are no spikes
        if n_spikes == 0:

            # find index of peak (absolute) response 
            if sweep_info['stim_amp'] < 0:
                peak_val_idx = np.argmin(v[start_idx:end_idx + 1]) + start_idx
            else:
                peak_val_idx = np.argmax(v[start_idx:end_idx + 1]) + start_idx

            # calculate membrane tau
            sweep_tau = calculate_membrane_tau(v, t, start_idx, peak_val_idx) * 1e3 # to ms
            
            analysis['subthresh'].append({'amp': float(sweep_info['stim_amp']), 
                                          'peak': float(v[peak_val_idx]),
                                          'peak_t': float(t[peak_val_idx]),
                                          'peak_idx': int(peak_val_idx),
                                          'steady': float(v[steady_interval_start_idx:end_idx].mean()),
                                          'tau': float(sweep_tau),
                                          'sweep_num': sweep_number,
                                          'base_vm': float(features.feature_list[-1].mean['base_v'])
                                          })
        elif sweep_info['stim_amp'] < rheo_amp:
            # there are spikes, and the stimulus amplitude 
            # is lower than the current rheobase amplitude
            rheo_n_spikes = n_spikes
            rheo_amp = sweep_info['stim_amp']
            rheo_sweep_num = sweep_number

        sweep_info['base_vm'] = float(features.feature_list[-1].mean['base_v'])

        analysis['sweep_info'].append(sweep_info)
        
        # Add point to the fI curve
        analysis['fI'].append((float(sweep_info['stim_amp']), float(n_spikes / sweep_info['stim_dur'])))
        

    if rheo_sweep_num < 0:
        logging.error("no rheo sweep found")
        return analysis

    try:
        analysis['input_resistance'] = float(calculate_input_resistance(analysis['subthresh']))
        analysis['tau'] = float(cellwide_tau(analysis['subthresh']))
    except ValueError:
        logging.error('input resistance calculation requires at least two passing, negative amplitude subthreshold long square sweeps.')
        return analysis

    analysis['v_rest'] = float(np.array([s['base_vm'] for s in analysis['sweep_info']]).mean())

    logging.info("Ri = {:.1f} MOhm".format(analysis['input_resistance']))
    logging.info("Tau_m = {:.1f} ms".format(analysis['tau']))
    logging.info("Vrest = {:.1f} mV".format(analysis['v_rest']))
    
    # Characterize the sag
    target = -100.0
    delta = 1e12
    sag_fraction = 0
    sag_vm = 0

    for s in analysis['subthresh']:
        if abs(s['peak'] - target) < delta:
            delta = abs(s['peak'] - target)
            sag_vm = s['peak']
            sag_fraction = (s['peak'] - s['steady']) / (s['peak'] - s['base_vm'])

    analysis['sag'] = float(sag_fraction)
    analysis['sag_eval_vm'] = float(sag_vm)

    # Get AP characteristics from rheobase sweep
    if rheo_sweep_num >= 0: 
        analysis['rheo_sweep_num'] = int(rheo_sweep_num)
        rheo_idx = [ i for i,sn in enumerate(sweep_numbers) if sn == rheo_sweep_num ][0]

        ft = features.feature_list[rheo_idx].mean

        v, i, t = get_sweep_from_nwb(nwb_file, rheo_sweep_num)
        
        # Average all spikes
        analysis['rheo_avg'] = {}
        analysis['rheo_avg']['up_down_ratio'] = float(ft['upstroke'] / -ft['downstroke'])
        analysis['rheo_avg']['trough'] = float(ft['f_trough'])
        analysis['rheo_avg']['fast_trough'] = float(ft['f_fast_ahp'])
        analysis['rheo_avg']['slow_trough'] = float(ft['f_slow_ahp'])
        analysis['rheo_avg']['has_true_ahp'] = bool(ft['f_trough'] < ft['base_v'])
        analysis['rheo_avg']['thresh_v'] = float(ft['threshold'])
        
        # First spike values
        analysis['rheo_spike_0'] = {}
        analysis['rheo_spike_0']['up_down_ratio'] = float(ft['spikes'][0]['upstroke'] / -ft['spikes'][0]['downstroke'])
        analysis['rheo_spike_0']['trough_v'] = float(ft['spikes'][0]['f_trough'])
        analysis['rheo_spike_0']['trough_t'] = float(ft['spikes'][0]['trough_t'] - t[start_idx])
        analysis['rheo_spike_0']['fast_trough_v'] = float(ft['spikes'][0]['f_fast_ahp'])
        analysis['rheo_spike_0']['fast_trough_t'] = float(ft['spikes'][0]['f_fast_ahp_t'] - t[start_idx])
        analysis['rheo_spike_0']['slow_trough_v'] = float(ft['spikes'][0]['f_slow_ahp'])
        analysis['rheo_spike_0']['slow_trough_t'] = float(ft['spikes'][0]['f_slow_ahp_t'] - t[start_idx])
        analysis['rheo_spike_0']['has_true_ahp'] = bool(ft['spikes'][0]['f_trough'] < ft['base_v'])
        analysis['rheo_spike_0']['thresh_v'] = float(ft['spikes'][0]['threshold'])
        analysis['rheo_spike_0']['thresh_t'] = float(ft['spikes'][0]['t'] - t[start_idx])
        analysis['rheo_spike_0']['thresh_i'] = float(i[ft['spikes'][0]['t_idx']])
        analysis['rheo_spike_0']['peak_v'] = float(ft['spikes'][0]['f_peak'])
        analysis['rheo_spike_0']['peak_t'] = float(ft['spikes'][0]['f_peak_t'] - t[start_idx])

    logging.info("Upstroke/downstroke ratio = {:.2f}".format(analysis['rheo_avg']['up_down_ratio']))
    
    # Get spike pattern characteristics from rheobase +40pA to +60pA. if none available, exit analysis.
    hero_idx = -1
    hero_range = [ 39.0, 61.0 ]

    # find hero sweeps within the range of valid pA offsets
    hero_sweeps = [ (s,i) for i,s in enumerate(analysis['sweep_info']) if (s['stim_amp'] - rheo_amp >= hero_range[0] and 
                                                                           s['stim_amp'] - rheo_amp <= hero_range[1]) ]

    if len(hero_sweeps) == 0:
        logging.error("could not find hero sweep, giving up on analysis.")
        return analysis
                
    # sort the potential hero sweeps in ascending order
    hero_sweeps.sort(key=lambda s: s[0]['stim_amp'])

    # grab the sweep with the lowest amplitude
    hero_idx = hero_sweeps[0][1]

    analysis['hero_sweep_num'] = sweep_numbers[hero_idx]
    ft = features.feature_list[hero_idx].mean

    spikes = ft['spikes']

    if 'isi_avg' in ft and ft['isi_avg'] is not None:
        avg_isi = ft['isi_avg']
        analysis['avg_isi'] = float(avg_isi)

    if 'adapt' in ft and ft['adapt'] is not None:
        analysis['adaptation_index'] = float(ft['adapt'])

    if 'latency' in ft:
        analysis['latency'] = float(ft['latency'])
    
    if len(spikes) >= 2:
        isis = [spikes[i + 1]['t'] - spikes[i]['t'] for i in range(len(spikes)-1)]

        if len(isis) > 0:
            analysis['has_delay'] = bool(ft['latency'] > avg_isi)
            analysis['delay_ratio'] = float(ft['latency'] / avg_isi)

        if len(isis) > 1:
            analysis['has_burst'] = bool(isis[0] <= 5e-3 and isis[1] <= 5e-3)
            analysis['isi_0'] = float(isis[0])
            analysis['isi_1'] = float(isis[1])

        if len(isis) > 2:
            analysis['has_pause'] = bool(check_for_pause(isis))
        
    # Figure out metric for fI curve shape - just fitting non-zero part to line
    fI_sorted = sorted(analysis['fI'], key=lambda d: d[0])
    x = np.array([d[0] for d in fI_sorted], dtype=np.float64)
    y = np.array([d[1] for d in fI_sorted], dtype=np.float64)
    last_zero_idx = np.nonzero(y)[0][0] - 1
    A = np.vstack([x[last_zero_idx:], np.ones(len(x[last_zero_idx:]))]).T
    m, c = np.linalg.lstsq(A, y[last_zero_idx:])[0]
    analysis['fI_fit_slope'] = float(m)
    
    logging.info("Linear fit to f-I curve slope = {:.2f}".format(analysis['fI_fit_slope']))

    return analysis
    

def analyze_ramps(sweep_numbers, nwb_file):
    '''
    Extract ramp-specific features from a set of sweeps in an NWB file.  These are
    primarily aggregate spike features.

    Parameters
    ----------

    sweep_numbers: list
        List of ramp sweep numbers.

    nwb_file: string
        File name of NWB file.
    '''

    analysis = {}
    analysis['sweep_info'] = []

    if len(sweep_numbers) == 0:
        return analysis
        
    features = fx.EphysFeatureExtractor()
    
    # Average values across ramps, since all ramps should have the same stimulus applied
    # and only look at the first spike
    sweep_count = 0
    value_keys = ['up_down_ratio', 'trough_v', 'trough_t', 'fast_trough_v', 'fast_trough_t', 'slow_trough_v', 'slow_trough_t', 'thresh_v', 'thresh_t', 'thresh_i', 'peak_t', 'peak_v']
    for k in value_keys:
        analysis[k] = 0
        
    for sweep_number in sorted(sweep_numbers):
        sweep_info = {}
        sweep_info['sweep_num'] = sweep_number

        v, i, t = get_sweep_from_nwb(nwb_file, sweep_number)
        
        if np.all(i == 0): # Check for weird ramp sweeps (orca conversion issue?)
            logging.warning("ramp sweep %d had all-zero stimulus.  skipping. " % sweep_number)
            continue
            
        sweep_info['stim_start'], start_idx = get_ramp_stim_characteristics(i, t)
        
        # Figure out where to end, since it goes to zero when cut off
        rev_v = v[::-1]
        nonzero_idx = np.flatnonzero(rev_v)[0]
        end_idx = len(v) - nonzero_idx - 1
            
        features.process_instance(sweep_number, v, i, t, 
                                  sweep_info['stim_start'], t[end_idx] - sweep_info['stim_start'], 
                                  'ramp')

        n_spikes = features.feature_list[-1].mean['n_spikes']
        
        
        if n_spikes > 0:
            sweep_count += 1

            spk = features.feature_list[-1].mean['spikes'][0]
            analysis['up_down_ratio'] += float(spk['upstroke'] / -spk['downstroke'])
            analysis['trough_v'] += float(spk['f_trough'])
            analysis['trough_t'] += float(spk['trough_t'] - t[start_idx])
            analysis['fast_trough_v'] += float(spk['f_fast_ahp'])
            analysis['fast_trough_t'] += float(spk['f_fast_ahp_t'] - t[start_idx])
            analysis['slow_trough_v'] += float(spk['f_slow_ahp'])
            analysis['slow_trough_t'] += float(spk['f_slow_ahp_t'] - t[start_idx])
            analysis['thresh_v'] += float(spk['threshold'])
            analysis['thresh_t'] += float(spk['t'] - t[start_idx])
            analysis['thresh_i'] += float(i[spk['t_idx']])
            analysis['peak_v'] += float(spk['f_peak'])
            analysis['peak_t'] += float(spk['f_peak_t'] - t[start_idx])

        analysis['sweep_info'].append(sweep_info)

    logging.info("Ramps with spikes: {:d}".format(sweep_count))

    if sweep_count > 0:
        for k in value_keys:
            analysis[k] /= sweep_count

        logging.info("Ramp up/down ratio: {:.1f}".format(analysis['up_down_ratio']))
        logging.info("Ramp threshold: {:.1f} mV at {:.1f} pA".format(analysis['thresh_v'], analysis['thresh_i']))
    else:
        for k in value_keys:
            analysis[k] = None

    analysis['sweep_count'] = sweep_count
    
    return analysis


def analyze_short_squares(sweep_numbers, nwb_file):
    '''
    Extract ramp-specific features from a set of sweeps in an NWB file.  Most
    of the features come from the lowest-amplitude spike-inducing sweeps.

    Parameters
    ----------

    sweep_numbers: list
        List of short square sweep numbers.

    nwb_file: string
        File name of NWB file.
    '''

    analysis = {}
    analysis['sweep_info'] = []

    if len(sweep_numbers) == 0:
        return analysis

    features = fx.EphysFeatureExtractor()
    spike_info = {}
    evoking_amps = []

    for sweep_number in sorted(sweep_numbers):
        sweep_info = {}
        sweep_info['sweep_num'] = sweep_number

        v, i, t = get_sweep_from_nwb(nwb_file, sweep_number)

        sweep_info['stim_start'], sweep_info['stim_dur'], sweep_info['stim_amp'], start_idx, end_idx = get_square_stim_characteristics(i, t)

        # We are actually looking for spikes *after* the end of the stimulus
        features.process_instance(sweep_number, v, i, t, 
                                  sweep_info['stim_start'], t[-1] - sweep_info['stim_start'],
                                  'short_square')

        n_spikes = features.feature_list[-1].mean['n_spikes']
        if n_spikes > 0:
            spike_info[sweep_number] = features.feature_list[-1].mean['spikes'][0]
            evoking_amps.append(sweep_info['stim_amp'])
        analysis['sweep_info'].append(sweep_info)
        
    # We want to analyze the smallest amplitude short_squares that have the most spikes evoked
    # So, if there is one 100pA short_square that evoked one spike, but there are 3 110pA short_squares
    # that evoke spikes, we'll analyze the 110pA ones.
    min_amp = 1e12
    min_count = 0
    amp_counts = Counter(evoking_amps)
    for amp in amp_counts:
        if amp_counts[amp] > min_count or (amp_counts[amp] == min_count and amp < min_amp):
            min_amp = amp
            min_count = amp_counts[amp]
    
    # now find the spikes at that amplitude and average their stats
    sweep_count = 0
    value_keys = ['up_down_ratio', 'trough_v', 'trough_t', 'fast_trough_v', 'fast_trough_t', 'slow_trough_v', 'slow_trough_t', 'thresh_v', 'thresh_t', 'thresh_i', 'peak_t', 'peak_v']

    for k in value_keys:
        analysis[k] = 0

    for si in analysis['sweep_info']:
        if si['stim_amp'] == min_amp and si['sweep_num'] in spike_info:
            spk = spike_info[si['sweep_num']]
            analysis['up_down_ratio'] += float(spk['upstroke'] / -spk['downstroke'])
            analysis['trough_v'] += float(spk['f_trough'])
            analysis['trough_t'] += float(spk['trough_t'] - t[start_idx])
            analysis['fast_trough_v'] += float(spk['f_fast_ahp'])
            analysis['fast_trough_t'] += float(spk['f_fast_ahp_t'] - t[start_idx])
            analysis['slow_trough_v'] += float(spk['f_slow_ahp'])
            analysis['slow_trough_t'] += float(spk['f_slow_ahp_t'] - t[start_idx])
            analysis['thresh_v'] += float(spk['threshold'])
            analysis['thresh_t'] += float(spk['t'] - si['stim_start'])
            analysis['thresh_i'] += float(si['stim_amp'])
            analysis['peak_v'] += float(spk['f_peak'])
            analysis['peak_t'] += float(spk['f_peak_t'] - t[start_idx])
            sweep_count += 1

    if sweep_count > 0:
        for k in value_keys:
            analysis[k] /= sweep_count
        analysis['repeat_amp'] = float(min_amp)
        logging.info("Short Square up/down ratio: {:.1f}".format(analysis['up_down_ratio']))
        logging.info("Short Square threshold: {:.1f} mV at {:.1f} pA".format(analysis['thresh_v'], analysis['thresh_i']))
    else:
        for k in value_keys:
            analysis[k] = None
    return analysis
    

def get_sweep_from_nwb(nwb_file, sweep_num):
    '''
    Read a sweep from an NWB file and convert Volts -> mV and Amps -> pA. 
    '''
    ds = NwbDataSet(nwb_file)
    data = ds.get_sweep(sweep_num)

    v = data['response'] * 1e3 # convert to mV
    i = data['stimulus'] * 1e12 # convert to pA

    dt = 1.0 / data['sampling_rate']
    t = np.arange(0,len(v)) * dt
    
    return (v, i, t)
        

def get_square_stim_characteristics(i, t, no_test_pulse=False):
    '''
    Identify the start time, duration, amplitude, start index, and
    end index of a square stimulus.  
    This assumes that there is a test pulse followed by the stimulus square.
    '''

    di = np.diff(i)
    up_idx = np.flatnonzero(di > 0)
    down_idx = np.flatnonzero(di < 0)

    idx = 0 if no_test_pulse else 1
    
    # second square is the stimulus
    if up_idx[idx] < down_idx[idx]: # positive square
        start_idx = up_idx[idx] + 1 # shift by one to compensate for diff()
        end_idx = down_idx[idx] + 1
    else: # negative square
        start_idx = down_idx[idx] + 1
        end_idx = up_idx[idx] + 1

    stim_start = float(t[start_idx])
    stim_dur = float(t[end_idx] - t[start_idx])
    stim_amp = float(i[start_idx])

    return (stim_start, stim_dur, stim_amp, start_idx, end_idx)


def get_ramp_stim_characteristics(i, t):
    ''' Identify the start time and start index of a ramp sweep. '''

    # Assumes that there is a test pulse followed by the stimulus ramp
    di = np.diff(i)
    up_idx = np.flatnonzero(di > 0)
    
    start_idx = up_idx[1] + 1 # shift by one to compensate for diff()
    return (t[start_idx], start_idx)
    

def get_stim_characteristics(i, t, no_test_pulse=False):
    '''
    Identify the start time, duration, amplitude, start index, and
    end index of a general stimulus.  
    This assumes that there is a test pulse followed by the stimulus square.
    '''

    di = np.diff(i)
    diff_idx = np.flatnonzero(di != 0)

    if len(diff_idx) == 0:
        return (None, None, 0.0, None, None)

    # skip the first up/down 
    idx = 0 if no_test_pulse else 1
    
    # shift by one to compensate for diff()
    start_idx = diff_idx[idx] + 1
    end_idx = diff_idx[-1] + 1

    stim_start = float(t[start_idx])
    stim_dur = float(t[end_idx] - t[start_idx])
    stim_amp = float(i[start_idx])

    return (stim_start, stim_dur, stim_amp, start_idx, end_idx)

def calculate_input_resistance(subthresh_data):
    ''' 
    Calculate the input resistance of a sweep using the output of a EphysFeatureExtractor instance 
    applied to long square sweeps.  This only uses the negative-going squares, and ignores
    sweeps that exceed -100 pA since some cells start to move from linear in that regime.
    '''

    i = np.array([d['amp'] for d in subthresh_data if d['amp'] < 0 and d['amp'] > -100])
    v = np.array([d['peak'] for d in subthresh_data if d['amp'] < 0 and d['amp'] > -100])
    A = np.vstack([i, np.ones(len(i))]).T
    m, c = np.linalg.lstsq(A, v)[0]
    return m * 1e3 # since MOhm = 1e3 * mV / pA


def cellwide_tau(subthresh_data):
    ''' Return the tau of the negative-going long square sweeps above -100 pA. '''

    taus = np.array([d['tau'] for d in subthresh_data if d['amp'] < 0 and d['amp'] > -100 and not np.isnan(d['tau'])])
    return taus.mean()


def calculate_membrane_tau(v, t, start_idx, peak_idx):
    ''' 
    Calculate membrane tau by fitting the response of the rheobase sweep.
    
    Parameters
    ----------

    v: np.ndarray
        array of voltages in mV

    t: np.ndarray
        array of time stamps in seconds

    start_index: int
        index of stimulus onset

    peak_idx: int
        index of the peak of the spike
    '''
    tenpct_idx, popt = fit_membrane_tau(v, t, start_idx, peak_idx)
    return 1.0 / popt[1] # seconds
    

def fit_membrane_tau(v, t, start_idx, peak_idx):
    ''' 
    Calculate membrane tau by fitting the response of the rheobase sweep.
    
    Parameters
    ----------

    v: np.ndarray
        array of voltages in mV

    t: np.ndarray
        array of time stamps in seconds

    start_index: int
        index of stimulus onset

    peak_idx: int
        index of the peak of the spike
    '''

    try:
        # fit from 10% up to peak
        if v[start_idx] > v[peak_idx]:
            tenpct_idx = np.where(v[start_idx:peak_idx] <= 0.1 * (v[peak_idx] - v[start_idx]) + v[start_idx])[0][0] + start_idx
        else:
            tenpct_idx = np.where(v[start_idx:peak_idx] >= 0.1 * (v[peak_idx] - v[start_idx]) + v[start_idx])[0][0] + start_idx

        guess = [v[tenpct_idx] - v[peak_idx], 50., v[peak_idx]]
        try:
            popt, pcov = curve_fit(exp_curve, t[tenpct_idx:peak_idx].astype(np.float64) - t[tenpct_idx], v[tenpct_idx:peak_idx].astype(np.float64), p0=guess)
        except RuntimeError:
            logging.warning("Curve fit for membrane tau failed.")
            return (np.nan, [np.nan, np.nan, np.nan])
        return (tenpct_idx, popt)
    except IndexError:
        logging.warning("Index error occurred calculating tau. Aborting calculation")
        return (np.nan, [np.nan, np.nan, np.nan])
    

def exp_curve(x, a, inv_tau, y0):
    ''' Function used for tau curve fitting '''
    return y0 + a * np.exp(-inv_tau * x)


def check_for_pause(isis):
    ''' Detect a pause given a list of inter-spike intervals. '''
    for i, isi in enumerate(isis[1:-1]):
        if isi > 3 * isis[i - 1 + 1] and isi > 3 * isis[i + 1 + 1]:
            return True
    return False


if __name__ == "__main__": pass


