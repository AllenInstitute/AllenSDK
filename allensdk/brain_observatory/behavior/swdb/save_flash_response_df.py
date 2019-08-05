import sys
import os
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

import behavior_project_cache as bpc
from importlib import reload; reload(bpc)

def get_nearest_frame(timepoint, timestamps):
    import bisect
    nearest_frame = bisect.bisect_left(timestamps, timepoint)
    return nearest_frame

def get_trace_around_timepoint(trace, timepoint, timestamps, window_around_timepoint_seconds, frame_rate):
    window = window_around_timepoint_seconds
    frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    lower_frame = frame_for_timepoint + int((window[0] * frame_rate))
    upper_frame = frame_for_timepoint + int((window[1] * frame_rate))
    trace = trace[lower_frame:upper_frame]
    timepoints = timestamps[lower_frame:upper_frame]
    return trace, timepoints

def get_mean_in_window(trace, window_after_trace_start_seconds, frame_rate):
    window = window_after_trace_start_seconds.copy()
    mean = np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)]) 
    return mean

def get_flash_response_df(session, response_analysis_params):
    frame_rate = 31.
    import itertools
    # get data to analyze
    dff_traces = session.dff_traces.copy()
    # for when no cell specimen id
    dff_traces.index = dff_traces.cell_roi_id.values
    flashes = session.stimulus_presentations.copy()
    # get params to define response window, in seconds
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    response_window_duration_seconds = response_analysis_params['response_window_duration_seconds']
    baseline_window_duration_seconds = response_analysis_params['baseline_window_duration_seconds']
    mean_response_window_seconds = [np.abs(window_around_timepoint_seconds[0]), 
                        np.abs(window_around_timepoint_seconds[0]) + response_window_duration_seconds]
    baseline_window_seconds = [np.abs(window_around_timepoint_seconds[0]) - baseline_window_duration_seconds, 
                        np.abs(window_around_timepoint_seconds[0])]
    cell_flash_combinations = itertools.product(dff_traces.index,flashes.index)
    index = pd.MultiIndex.from_tuples(cell_flash_combinations, names=['cell_specimen_id', 'flash_id'])
    df = pd.DataFrame(index=index)
    traces_list = []
    trace_timestamps_list = []
    for cell_specimen_id, flash_id in itertools.product(dff_traces.index,flashes.index):
        timepoint = flashes.loc[flash_id]['start_time']
        cell_roi_id = dff_traces.loc[cell_specimen_id]['cell_roi_id']
        full_cell_trace = dff_traces.loc[cell_specimen_id, 'dff']
        trace, trace_timestamps = get_trace_around_timepoint(full_cell_trace, timepoint, session.ophys_timestamps, window_around_timepoint_seconds, frame_rate)
        mean_response = get_mean_in_window(trace, mean_response_window_seconds, frame_rate)
        baseline_response = get_mean_in_window(trace, baseline_window_seconds, frame_rate) 
        traces_list.append(trace.tolist())
        trace_timestamps_list.append(trace_timestamps.tolist())
        df.loc[(cell_specimen_id, flash_id), 'cell_roi_id'] = int(cell_roi_id)
        df.loc[(cell_specimen_id, flash_id), 'mean_response'] = mean_response
        df.loc[(cell_specimen_id, flash_id), 'baseline_response'] = baseline_response
    df.insert(loc=1, column='dff_trace', value=traces_list)
    df.insert(loc=2, column='dff_trace_timestamps', value=trace_timestamps_list)
    return df

# window_around_timepoint_seconds:  window around each timepoint to take a snippet of the dF/F trace
# response_window_duration_seconds: time in seconds after timepoint over which to take the mean response
# baseline_window_duration_seconds: time in seconds before timepoint over which to take the baseline response

if __name__=='__main__':

    experiment_id = sys.argv[1]
    cache_json = {'manifest_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/visual_behavior_data_manifest.csv',
                  'nwb_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/nwb_files',
                  'analysis_files_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files'
                  }

    cache = bpc.BehaviorProjectCache(cache_json)
    #  experiment_id = cache.manifest.iloc[5]['ophys_experiment_id']
    nwb_path = cache.get_nwb_filepath(experiment_id)
    api = BehaviorOphysNwbApi(nwb_path)
    session = BehaviorOphysSession(api)

    output_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files'

    response_analysis_params = {'window_around_timepoint_seconds':[-.5,.75], # -500ms, 750ms
                               'response_window_duration_seconds': 0.75, 
                               'baseline_window_duration_seconds': 0.5} 

    flash_response_df = get_flash_response_df(session, response_analysis_params)

    output_fn = os.path.join(output_path, 'flash_response_df_{}.h5'.format(experiment_id))
    print('Writing flash response df to {}'.format(output_fn))
    flash_response_df.to_hdf(output_fn, key='df')


