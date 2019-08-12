import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
from importlib import reload; reload(bpc)
from allensdk.brain_observatory.behavior.swdb.analysis_tools import get_nearest_frame, get_trace_around_timepoint, get_mean_in_window


def add_p_vals_tr(tr):
    tr['p_value'] = 1.
    response_window = [4, 4.5]  
    ophys_frame_rate=31.
    for index,row in tr.iterrows():
        tr.at[index,'p_value'] = get_p_val(row.dff_trace, response_window, ophys_frame_rate)
    return tr

def get_p_val(trace, response_window, frame_rate):
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p

def annotate_trial_response_df_with_pref_stim(trial_response_df):
    rdf = trial_response_df.copy()
    rdf['pref_stim'] = False
    cell_key = 'cell_specimen_id'
    mean_response = rdf.groupby([cell_key, 'change_image_name']).apply(get_mean_sem_trace)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values == np.max(m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        rdf = rdf.reset_index()
        rdf = rdf.set_index(['cell_specimen_id','change_image_name'])
        rdf.at[(cell,pref_image),'pref_stim'] = True
    rdf = rdf.reset_index()
    rdf = rdf.set_index(['cell_specimen_id','trial_id'])
    return rdf

def get_mean_sem_trace(group):
    '''
        Computes the average and sem of the mean_response column
    '''
    mean_response = np.mean(group['mean_response'])
    mean_responses = group['mean_response'].values
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    mean_trace = np.mean(group['dff_trace'])
    sem_trace = np.std(group['dff_trace'].values) / np.sqrt(len(group['dff_trace'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace,
                      'mean_responses': mean_responses})

def get_trial_response_df(session, response_analysis_params):
    frame_rate = 31.
    import itertools
    # get data to analyze
    dff_traces = session.dff_traces.copy()
    # for when no cell specimen id
    #  dff_traces.index = dff_traces.cell_roi_id.values
    trials = session.trials.copy()
    trials = trials[trials.aborted==False]

    # get params to define response window, in seconds
    window_around_timepoint_seconds = response_analysis_params['window_around_timepoint_seconds']
    response_window_duration_seconds = response_analysis_params['response_window_duration_seconds']
    baseline_window_duration_seconds = response_analysis_params['baseline_window_duration_seconds']
    mean_response_window_seconds = [np.abs(window_around_timepoint_seconds[0]), 
                        np.abs(window_around_timepoint_seconds[0]) + response_window_duration_seconds]
    baseline_window_seconds = [np.abs(window_around_timepoint_seconds[0]) - baseline_window_duration_seconds, 
                        np.abs(window_around_timepoint_seconds[0])]

    cell_trial_combinations = itertools.product(dff_traces.index,trials.index)
    index = pd.MultiIndex.from_tuples(cell_trial_combinations, names=['cell_specimen_id', 'trial_id'])
    df = pd.DataFrame(index=index)

    traces_list = []
    trace_timestamps_list = []
    for cell_specimen_id, trial_id in itertools.product(dff_traces.index,trials.index):
        timepoint = trials.loc[trial_id]['change_time']
        cell_roi_id = dff_traces.loc[cell_specimen_id]['cell_roi_id']
        full_cell_trace = dff_traces.loc[cell_specimen_id, 'dff']
        trace, trace_timestamps = get_trace_around_timepoint(full_cell_trace, timepoint, session.ophys_timestamps, 
                                                             window_around_timepoint_seconds, frame_rate) #session.metadata['ophys_frame_rate'])
        mean_response = get_mean_in_window(trace, mean_response_window_seconds, frame_rate) #session.metadata['ophys_frame_rate'])
        baseline_response = get_mean_in_window(trace, baseline_window_seconds, frame_rate) #session.metadata['ophys_frame_rate'])

        traces_list.append(trace)
        trace_timestamps_list.append(trace_timestamps)
        df.loc[(cell_specimen_id, trial_id), 'cell_roi_id'] = int(cell_roi_id)
        df.loc[(cell_specimen_id, trial_id), 'mean_response'] = mean_response
        df.loc[(cell_specimen_id, trial_id), 'baseline_response'] = baseline_response
    df.insert(loc=1, column='dff_trace', value=traces_list)
    df.insert(loc=2, column='dff_trace_timestamps', value=trace_timestamps_list)
    return df

# window_around_timepoint_seconds: window around each timepoint to take a snippet of the dF/F trace
# response_window_duration_seconds: time in seconds after timepoint over which to take the mean response
# baseline_window_duration_seconds: time in seconds before timepoint over which to take the baseline response

if __name__=='__main__':
    cache_json = {'manifest_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/visual_behavior_data_manifest.csv',
                  'nwb_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/nwb_files',
                  'analysis_files_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files_final'
                  }

    cache = bpc.BehaviorProjectCache(cache_json)

    case=0

    if case == 0:

        experiment_id = sys.argv[1]
        #  experiment_id = cache.manifest.iloc[5]['ophys_experiment_id']
        nwb_path = cache.get_nwb_filepath(experiment_id)
        #  api = BehaviorOphysNwbApi(nwb_path, filter_invalid_rois=True)
        #  session = BehaviorOphysSession(api)

        # Get the session using the cache so that the change time fix is applied
        session = cache.get_session(experiment_id)
        change_times = session.trials['change_time'][~pd.isnull(session.trials['change_time'])].values
        flash_times = session.stimulus_presentations['start_time'].values
        assert np.all(np.isin(change_times, flash_times))

        output_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files_final'

        response_analysis_params = {'window_around_timepoint_seconds':[-4,8],
                                   'response_window_duration_seconds': 0.5,
                                   'baseline_window_duration_seconds': 0.5}

        trial_response_df = get_trial_response_df(session, response_analysis_params)
        trial_response_df = add_p_vals_tr(trial_response_df)
        #  trial_response_df = annotate_trial_response_df_with_pref_stim(trial_response_df)

        output_fn = os.path.join(output_path, 'trial_response_df_{}.h5'.format(experiment_id))
        print('Writing trial response df to {}'.format(output_fn))
        trial_response_df.to_hdf(output_fn, key='df', complib='bzip2', complevel=9)

    elif case == 1: 
        experiment_id  = 846487947

        #  api = BehaviorOphysLimsApi(experiment_id)
        #  session = BehaviorOphysSession(api)
        #  nwb_path = cache.get_nwb_filepath(experiment_id)
        #  api = BehaviorOphysNwbApi(nwb_path)
        #  session = BehaviorOphysSession(api)

        session = cache.get_session(experiment_id)

        change_times = session.trials['change_time'][~pd.isnull(session.trials['change_time'])].values
        flash_times = session.stimulus_presentations['start_time'].values
        assert np.all(np.isin(change_times, flash_times))

        output_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files_final'

        response_analysis_params = {'window_around_timepoint_seconds':[-4,8],
                                   'response_window_duration_seconds': 0.5,
                                   'baseline_window_duration_seconds': 0.5}

        trial_response_df = get_trial_response_df(session, response_analysis_params)

        trial_metadata = session.trials.copy()
        trial_metadata.index.names = ['trial_id']
        trial_response_df = trial_response_df.join(trial_metadata)
        trial_response_df = add_p_vals_tr(trial_response_df)
        trial_response_df = annotate_trial_response_df_with_pref_stim(trial_response_df)

