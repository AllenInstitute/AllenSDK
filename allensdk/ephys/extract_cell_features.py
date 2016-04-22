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

import logging

from allensdk.core.nwb_data_set import NwbDataSet

import allensdk.ephys.ephys_extractor as efex
import allensdk.ephys.ephys_features as ft

HERO_MIN_AMP_OFFSET = 39.0
HERO_MAX_AMP_OFFSET = 61.0

SHORT_SQUARE_TYPES = ["Short Square", 
                      "Short Square - Triple", 
                      "Short Square - Hold -60mv", 
                      "Short Square - Hold -70mv", 
                      "Short Square - Hold -80mv"]

SHORT_SQUARE_MAX_THRESH_FRAC = 0.1

MEAN_FEATURES = [ "upstroke_downstroke_ratio", "peak_v", "peak_t", "trough_v", "trough_t",
                  "fast_trough_v", "fast_trough_t", "slow_trough_v", "slow_trough_t",
                  "threshold_v", "threshold_i", "threshold_t", "peak_v", "peak_t" ]


def extract_sweep_features(data_set, sweeps_by_type):
    # extract sweep-level features
    sweep_features = {}

    for stimulus_type, sweep_numbers in sweeps_by_type.iteritems():
        logging.debug("%s:%s" % (stimulus_type, ','.join(map(str, sweep_numbers))))

        if stimulus_type in SHORT_SQUARE_TYPES:
            tmp_ext = efex.extractor_for_nwb_sweeps(data_set, sweep_numbers)
            t_set = [s.t for s in tmp_ext.sweeps()]
            v_set = [s.v for s in tmp_ext.sweeps()]

            cutoff, thresh_frac = ft.estimate_adjusted_detection_parameters(v_set, t_set,
                                                                            efex.SHORT_SQUARES_WINDOW_START,
                                                                            efex.SHORT_SQUARES_WINDOW_END)
            thresh_frac = max(SHORT_SQUARE_MAX_THRESH_FRAC, thresh_frac)

            fex = efex.extractor_for_nwb_sweeps(data_set, sweep_numbers,
                                                dv_cutoff=cutoff, thresh_frac=thresh_frac)
        else:
            fex = efex.extractor_for_nwb_sweeps(data_set, sweep_numbers)

        fex.process_spikes()
        
        sweep_features.update({ f.id:f.as_dict() for f in fex.sweeps() })            

    return sweep_features

def extract_cell_features(data_set,
                          ramp_sweep_numbers, 
                          short_square_sweep_numbers, 
                          long_square_sweep_numbers):

    fex = efex.cell_extractor_for_nwb(data_set, 
                                      ramp_sweep_numbers,
                                      short_square_sweep_numbers,
                                      long_square_sweep_numbers)

    fex.process()

    cell_features = fex.as_dict()

    # find hero sweep
    rheo_amp = cell_features['long_squares']['rheobase_i']
    hero_min, hero_max = rheo_amp + HERO_MIN_AMP_OFFSET, rheo_amp + HERO_MAX_AMP_OFFSET
    hero_amp = float("inf")
    hero_sweep = None
    for sweep in fex.long_squares_features("spiking").sweeps():
        nspikes = len(sweep.spikes())
        amp = sweep.sweep_feature("stim_amp")
        
        if nspikes > 0 and amp > hero_min and amp < hero_max and amp < hero_amp:
            hero_amp = amp
            hero_sweep = sweep

    if hero_sweep:
        adapt = hero_sweep.sweep_feature("adapt")
        latency = hero_sweep.sweep_feature("latency")
        mean_isi = hero_sweep.sweep_feature("mean_isi")
    else:
        raise ft.FeatureError("Could not find hero sweep.")  

    # find the mean features of the first spike for the ramps and short squares
    ramps_ms0 = mean_features_spike_zero(fex.ramps_features().sweeps())
    ss_ms0 = mean_features_spike_zero(fex.short_squares_features().sweeps())

    # compute baseline from all long square sweeps
    v_baseline = np.mean(fex.long_squares_features().sweep_features('v_baseline'))

    cell_features['long_squares']['v_baseline'] = v_baseline
    cell_features['long_squares']['hero_sweep'] = hero_sweep.as_dict() if hero_sweep else None
    cell_features["ramps"]["mean_spike_0"] = ramps_ms0
    cell_features["short_squares"]["mean_spike_0"] = ss_ms0

    return cell_features
    
def mean_features_spike_zero(sweeps):
    """ Compute mean feature values for the first spike in list of extractors """

    output = {}
    for mf in MEAN_FEATURES:
        mfd = [ sweep.spikes()[0][mf] for sweep in sweeps if sweep.sweep_feature("avg_rate") > 0 ]
        output[mf] = np.mean(mfd)
    return output

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

def get_ramp_stim_characteristics(i, t):
    ''' Identify the start time and start index of a ramp sweep. '''

    # Assumes that there is a test pulse followed by the stimulus ramp
    di = np.diff(i)
    up_idx = np.flatnonzero(di > 0)
    
    start_idx = up_idx[1] + 1 # shift by one to compensate for diff()
    return (t[start_idx], start_idx)

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

if __name__ == "__main__": pass


