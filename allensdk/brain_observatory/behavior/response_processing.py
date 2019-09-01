## For calculating trial and flash responses
import sys
import os
import numpy as np
import math
import pandas as pd
from scipy import stats
import itertools
import xarray as xr

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

from allensdk.brain_observatory.behavior import behavior_project_cache as bpc
from allensdk.brain_observatory.behavior.swdb.analysis_tools import get_nearest_frame, get_trace_around_timepoint, get_mean_in_window

OPHYS_FRAME_RATE = 31.

trial_response_params = {
    "window_around_timepoint_seconds":[-4, 8],
    "response_window_duration_seconds":0.5,
    "baseline_window_duration_seconds":0.5
}

flash_response_params = {
    "window_around_timepoint_seconds":[-0.5, 0.75],
    "response_window_duration_seconds":0.5,
    "baseline_window_duration_seconds":0.5
}

def index_of_nearest_value(sample_times, event_times):
    '''
    The index of the nearest sample time for each event time.
    Args: 
        sample_times (np.ndarray of float): sorted 1-d vector of sample timestamps
        event_times (np.ndarray of float): 1-d vector of event timestamps
    Returns
        (np.ndarray of int) nearest sample time index for each event time
    '''
    insertion_ind = np.searchsorted(sample_times, event_times)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = sample_times[insertion_ind] - event_times
    ind_minus_one_diff = np.abs(sample_times[np.clip(insertion_ind-1, 0, np.inf).astype(int)] - event_times)
    return insertion_ind - (ind_diff>ind_minus_one_diff).astype(int)

def eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset):
    '''
    Extract trace for each cell, for each event-relative window.
    Args: 
        dff_traces (np.ndarray): shape (nSamples, nCells) with dff traces for each cell
        event_indices (np.ndarray): 1-d array of shape (nEvents) with closest sample ind for each event
        start_ind_offset (int): Where to start the window relative to each event ind
        end_ind_offset (int): Where to end the window relative to each event ind
    Returns:
        sliced_dataout (np.ndarray): shape (nSamples, nEvents, nCells)
    '''
    all_inds = event_indices + np.arange(start_ind_offset, end_ind_offset)[:,None] 
    sliced_dataout = dff_traces_arr.T[all_inds]
    return sliced_dataout
 
def trial_response_df(session, response_analysis_params=trial_response_params):
    dff_traces_arr = np.stack(session.dff_traces['dff'].values)
    change_trials = session.trials[~pd.isnull(session.trials['change_time'])]
    event_times = change_trials['change_time'].values
    event_indices = index_of_nearest_value(session.ophys_timestamps, event_times)
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * OPHYS_FRAME_RATE
    start_ind_offset = int(window_around_timepoint_seconds[0] * OPHYS_FRAME_RATE)
    end_ind_offset = int(start_ind_offset + trace_len)
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / OPHYS_FRAME_RATE
    sliced_dataout = eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset)
    result = xr.DataArray(
        data = sliced_dataout,
        dims = ("eventlocked_timestamps", "trials_id", "cell_specimen_id"),
        coords = {
            "eventlocked_timestamps": trace_timebase,
            "trials_id": change_trials.index.values,
            "cell_specimen_id": session.cell_specimen_table.index.values
        }
    )
    return result

def flash_response_df(session, response_analysis_params=flash_response_params):
    dff_traces_arr = np.stack(session.dff_traces['dff'].values)
    event_times = session.stimulus_presentations['start_time'].values
    event_indices = index_of_nearest_value(session.ophys_timestamps, event_times)
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * OPHYS_FRAME_RATE
    start_ind_offset = int(window_around_timepoint_seconds[0] * OPHYS_FRAME_RATE)
    end_ind_offset = int(start_ind_offset + trace_len)
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / OPHYS_FRAME_RATE
    sliced_dataout = eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset)
    result = xr.DataArray(
        data = sliced_dataout,
        dims = ("eventlocked_timestamps", "stimulus_presentations_id", "cell_specimen_id"),
        coords = {
            "eventlocked_timestamps": trace_timebase,
            "stimulus_presentations_id": session.stimulus_presentations.index.values,
            "cell_specimen_id": session.cell_specimen_table.index.values
        }
    )
    return result

def mean_response(eventlocked_response_xr):
    '''
    Calculate mean response and baseline for each neuron over events
    '''
    #  response_window_duration_seconds = response_analysis_params['response_window_duration_seconds']
    #  baseline_window_duration_seconds = response_analysis_params['baseline_window_duration_seconds']
    pass

if __name__=="__main__":
    import time
    manifest_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/behavior_project_manifest.json'
    cache = bpc.BehaviorProjectCache.fixed(manifest=manifest_path)
    session = cache.get_session_data(880961028)
    result = trial_response_df(session)
