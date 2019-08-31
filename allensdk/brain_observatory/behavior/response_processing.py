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

def get_trial_response_df(session, response_analysis_params):
    '''
        Computes the trial_response_df for the session
        PROBLEM: Ignores aborted trials    

        INPUTS:
        session, a behaviorOphysSession object to be analyzed
        response_analysis_params, a dictionary with keys:
            'window_around_timepoint_seconds'   The window around the change_time to use in the dff trace
            'response_window_duration_seconds'  The duration after the change time to use in the mean_response
            'baseline_window_duration_seconds'  The duration  before the change time to use as the baseline_response
    
        OUTPUTS:
        trial_response_df, a pandas dataframe with multi-index (cell_specimen_id/trial_id), and columns:
            cell_roi_id, this sessions roi_id
            mean_response, the average dff in the response_window
            baseline_response, the average dff in the baseline window
            dff_trace, the dff_trace in the window_around_timepoint_seconds
            dff_trace_timestamps, the timestamps for the dff_trace
    '''
    frame_rate = 31. #PROBLEM, shouldnt hard code this here

    # get data to analyze
    dff_traces = session.dff_traces.copy()
    trials = session.trials.copy()
    trials = trials[trials.aborted==False] # PROBLEM

    # get params to define response window, in seconds
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    response_window_duration_seconds = response_analysis_params['response_window_duration_seconds']
    baseline_window_duration_seconds = response_analysis_params['baseline_window_duration_seconds']
    mean_response_window_seconds = [np.abs(window_around_timepoint_seconds[0]), 
                        np.abs(window_around_timepoint_seconds[0]) + response_window_duration_seconds]
    baseline_window_seconds = [np.abs(window_around_timepoint_seconds[0]) - baseline_window_duration_seconds, 
                        np.abs(window_around_timepoint_seconds[0])]

    # Set up multi-index dataframe
    cell_trial_combinations = itertools.product(dff_traces.index,trials.index)
    index = pd.MultiIndex.from_tuples(cell_trial_combinations, names=['cell_specimen_id', 'trial_id'])
    df = pd.DataFrame(index=index)

    # Iterate through cell/trial pairs, and construct the columns
    traces_list = []
    trace_timestamps_list = []
    for cell_specimen_id, trial_id in itertools.product(dff_traces.index,trials.index):
        timepoint = trials.loc[trial_id]['change_time']
        cell_roi_id = dff_traces.loc[cell_specimen_id]['cell_roi_id']
        full_cell_trace = dff_traces.loc[cell_specimen_id, 'dff']
        trace, trace_timestamps = get_trace_around_timepoint(full_cell_trace, timepoint, session.ophys_timestamps, 
                                                             window_around_timepoint_seconds, frame_rate) 
        mean_response = get_mean_in_window(trace, mean_response_window_seconds, frame_rate) 
        baseline_response = get_mean_in_window(trace, baseline_window_seconds, frame_rate) 

        traces_list.append(trace)
        trace_timestamps_list.append(trace_timestamps)
        df.loc[(cell_specimen_id, trial_id), 'cell_roi_id'] = int(cell_roi_id)
        df.loc[(cell_specimen_id, trial_id), 'mean_response'] = mean_response
        df.loc[(cell_specimen_id, trial_id), 'baseline_response'] = baseline_response
    df.insert(loc=1, column='dff_trace', value=traces_list)
    df.insert(loc=2, column='dff_trace_timestamps', value=trace_timestamps_list)
    return df

def better_get_trial_response_df(session, response_analysis_params):
    
    OPHYS_FRAME_RATE = 31.
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    response_window_duration_seconds = response_analysis_params['response_window_duration_seconds']
    baseline_window_duration_seconds = response_analysis_params['baseline_window_duration_seconds']

    # How many samples are we going to pull? 
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * OPHYS_FRAME_RATE

    # Where do we start and end relative to the change time? 
    start_ind_offset = window_around_timepoint_seconds[0] * OPHYS_FRAME_RATE
    end_ind_offset = start_ind_offset + trace_len

    # What are the ophys sample inds closest to each change time?
    change_times = session.trials[~pd.isnull(session.trials['change_time'])]['change_time'].values
    


if __name__=="__main__":
    import time
    manifest_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/behavior_project_manifest.json'
    cache = bpc.BehaviorProjectCache.fixed(manifest=manifest_path)
    session = cache.get_session_data(880961028)

    def nearest_index(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    now=time.time()
    #  trial_response_df = get_trial_response_df(session, trial_response_params)
    #  elapsed = time.time() - now
    #  print(elapsed)

    #####
    #  response_analysis_params = trial_response_params
    response_analysis_params = flash_response_params

    OPHYS_FRAME_RATE = 31.
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    response_window_duration_seconds = response_analysis_params['response_window_duration_seconds']
    baseline_window_duration_seconds = response_analysis_params['baseline_window_duration_seconds']

    # How many samples are we going to pull? 
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * OPHYS_FRAME_RATE

    # Where do we start and end relative to the change time? 
    start_ind_offset = int(window_around_timepoint_seconds[0] * OPHYS_FRAME_RATE)
    end_ind_offset = int(start_ind_offset + trace_len)

    # What are the ophys sample inds closest to each change time?
    #  change_times = session.trials[~pd.isnull(session.trials['change_time'])]['change_time'].values
    event_times = session.stimulus_presentations['start_time'].values
    event_inds = session.stimulus_presentations.index.values

    # This is the ind of the actual closest time instead of just searchsorting
    event_inds_closest = np.array([
        nearest_index(session.ophys_timestamps, event_time)
        for event_time in event_times
    ])

    # TODO: Assert 0+start_ind_offset... etc
    assert np.all((event_inds_closest > 0)&(event_inds_closest<len(session.ophys_timestamps)))

    all_inds = event_inds_closest + np.arange(start_ind_offset, end_ind_offset)[:,None] 
    data = np.stack(session.dff_traces['dff'].values)
    sliced_dataout = data.T[all_inds]

    # NOTE: The zero here isn't the event time exactly, but the insertion ind...
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / OPHYS_FRAME_RATE

    cell_specimen_ids = session.cell_specimen_table.index.values

    result = xr.DataArray(
        data = sliced_dataout,
        dims = ("eventlocked_samples", "event_inds", "cell_specimen_id"),
        coords = {
            "eventlocked_samples": trace_timebase,
            "event_inds": event_inds,
            "cell_specimen_id": cell_specimen_ids
        }
    )

    elapsed = time.time() - now
    print(elapsed)
