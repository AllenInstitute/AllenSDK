import sys
import os
import numpy as np
import pandas as pd
import itertools

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
from allensdk.brain_observatory.behavior.swdb.analysis_tools import get_nearest_frame, get_trace_around_timepoint, get_mean_in_window

def get_flash_response_df(session, response_analysis_params):
    frame_rate = 31.
    # get data to analyze
    dff_traces = session.dff_traces.copy()
    # for when no cell specimen id
    #  dff_traces.index = dff_traces.cell_roi_id.values
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
        traces_list.append(trace)
        trace_timestamps_list.append(trace_timestamps)
        df.loc[(cell_specimen_id, flash_id), 'cell_roi_id'] = int(cell_roi_id)
        df.loc[(cell_specimen_id, flash_id), 'mean_response'] = mean_response
        df.loc[(cell_specimen_id, flash_id), 'baseline_response'] = baseline_response
    df.insert(loc=1, column='dff_trace', value=traces_list)
    df.insert(loc=2, column='dff_trace_timestamps', value=trace_timestamps_list)
    return df

def get_p_values_from_shuffled_spontaneous(session, flash_response_df, response_window_duration=0.5):
    '''
        Computes the P values for each cell/flash. The P value is the probability of observing a response of that
        magnitude in the spontaneous window. The algorithm is copied from VBA
    '''
    # Organize Data
    fdf = flash_response_df.copy()
    st  = session.stimulus_presentations.copy()
    included_flashes = fdf.index.get_level_values(1).unique()
    st  = st[st.index.isin(included_flashes)]
    # Get Sample of Spontaneous Frames
    spontaneous_frames = get_spontaneous_frames(session)
    #  Set up Params
    ophys_frame_rate = 31
    n_mean_response_window_frames = int(np.round(response_window_duration * ophys_frame_rate, 0))
    cell_ids = np.unique(fdf.index.get_level_values(0))    
    n_cells = len(cell_ids)
    # Get Shuffled responses from spontaneous frames
    # get mean response for 10000 shuffles of the spontaneous activity frames
    # in a window the same size as the stim response window duration
    shuffled_responses = np.empty((n_cells, 10000, n_mean_response_window_frames))
    idx = np.random.choice(spontaneous_frames, 10000)
    dff_traces = np.stack(session.dff_traces.to_numpy()[:,1],axis=0)
    for i in range(n_mean_response_window_frames):
        shuffled_responses[:, :, i] = dff_traces[:, idx + i]
    shuffled_mean = shuffled_responses.mean(axis=2)
    # compare flash responses to shuffled values and make a dataframe of p_value for cell by sweep
    #flash_p_values = pd.DataFrame(index=st.index.values, columns=cell_ids.astype(str))
    iterables = [cell_ids, st.index.values]
    flash_p_values = pd.DataFrame(index=pd.MultiIndex.from_product(iterables,names= ['cell_specimen_id','flash_id']))
    for i, cell_index in enumerate(cell_ids):
        responses = fdf.loc[cell_index].mean_response.values
        null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(responses), 1))
        actual_is_less = responses.reshape(len(responses), 1) <= null_dist_mat
        p_values = np.mean(actual_is_less, axis=1)
        for j in range(0,len(p_values)):
            flash_p_values.at[(cell_index,j),'p_value'] = p_values[j] 
    fdf = pd.concat([fdf,flash_p_values],axis=1)
    return fdf 

def get_spontaneous_frames(session):
    '''
        Returns a list of the frames that occur during the before and after spontaneous windows
    '''
    st = session.stimulus_presentations.copy()
    # dont use full 5 mins to avoid fingerprint and countdown
    # spont_duration_frames = 4 * 60 * 60  # 4 mins * * 60s/min * 60Hz
    spont_duration = 4 * 60  # 4mins * 60sec
    # for spontaneous at beginning of session
    behavior_start_time = st.iloc[0].start_time
    spontaneous_start_time_pre = behavior_start_time - spont_duration
    spontaneous_end_time_pre = behavior_start_time
    spontaneous_start_frame_pre = get_successive_frame_list(spontaneous_start_time_pre, session.ophys_timestamps)
    spontaneous_end_frame_pre = get_successive_frame_list(spontaneous_end_time_pre, session.ophys_timestamps)
    spontaneous_frames_pre = np.arange(spontaneous_start_frame_pre, spontaneous_end_frame_pre, 1)
    # for spontaneous epoch at end of session
    behavior_end_time = st.iloc[-1].stop_time
    spontaneous_start_time_post = behavior_end_time + 0.5
    spontaneous_end_time_post = behavior_end_time + spont_duration
    spontaneous_start_frame_post = get_successive_frame_list(spontaneous_start_time_post, session.ophys_timestamps)
    spontaneous_end_frame_post = get_successive_frame_list(spontaneous_end_time_post, session.ophys_timestamps)
    spontaneous_frames_post = np.arange(spontaneous_start_frame_post, spontaneous_end_frame_post, 1)
    # add them together
    spontaneous_frames = list(spontaneous_frames_pre) + (list(spontaneous_frames_post))
    return spontaneous_frames

def get_successive_frame_list(timepoints_array, timestamps):
    '''
        Returns the next frame after timestamps in timepoints_array
        copied from VBA
    '''
    # This is a modification of get_nearest_frame for speedup
    #  This implementation looks for the first 2p frame consecutive to the stim
    successive_frames = np.searchsorted(timestamps, timepoints_array)
    return successive_frames

def add_image_name(session,fdf):
    '''
        Adds a column to flash_response_df with the image_name taken from the stimulus_presentations table
        Slow to run
    '''
    fdf = fdf.reset_index()
    fdf = fdf.set_index('flash_id')
    fdf['image_name']= ''
    # So slow!!!
    for stim_id in np.unique(fdf.index.values):
        fdf.loc[stim_id,'image_name'] = session.stimulus_presentations.loc[stim_id].image_name
    fdf = fdf.reset_index()
    fdf = fdf.set_index(['cell_specimen_id','flash_id'])
    return fdf

def annotate_flash_response_df_with_pref_stim(fdf):
    '''
        Adds a column to flash_response_df with a boolean value of whether that flash was that cells pref image.
        Computes preferred image by looking for the image that on average evokes the largest response.
        Slow to run
    '''
    fdf = fdf.reset_index()
    if 'cell_specimen_id' in fdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    fdf['pref_stim'] = False
    mean_response = fdf.groupby([cell_key, 'image_name']).apply(get_mean_sem)
    m = mean_response.unstack()
    for cell in m.index:
        temp = np.where(m.loc[cell]['mean_response'].values == np.nanmax(m.loc[cell]['mean_response'].values))[0]
        if len(temp) > 0:
            image_index = temp[0]
            pref_image = m.loc[cell]['mean_response'].index[image_index]
            trials = fdf[(fdf[cell_key] == cell) & (fdf.image_name == pref_image)].index
            fdf.loc[trials, 'pref_stim'] = True
    fdf = fdf.set_index(['cell_specimen_id','flash_id'])
    return fdf

def get_mean_sem(group):
    '''
        Returns the mean and sem of the mean_response values for all entries in the group
    '''
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response})

# window_around_timepoint_seconds:  window around each timepoint to take a snippet of the dF/F trace
# response_window_duration_seconds: time in seconds after timepoint over which to take the mean response
# baseline_window_duration_seconds: time in seconds before timepoint over which to take the baseline response

if __name__=='__main__':

    case = 0

    if case==0:

        experiment_id = sys.argv[1]
        cache_json = {'manifest_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/visual_behavior_data_manifest.csv',
                      'nwb_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/nwb_files',
                      'analysis_files_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files'
                      }

        cache = bpc.BehaviorProjectCache(cache_json)
        #  experiment_id = cache.manifest.iloc[5]['ophys_experiment_id']
        nwb_path = cache.get_nwb_filepath(experiment_id)
        api = BehaviorOphysNwbApi(nwb_path, filter_invalid_rois = True)
        session = BehaviorOphysSession(api)

        output_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/extra_files_final'

        response_analysis_params = {'window_around_timepoint_seconds':[-.5,.75], # -500ms, 750ms
                                   'response_window_duration_seconds': 0.75, 
                                   'baseline_window_duration_seconds': 0.5} 

        flash_response_df = get_flash_response_df(session, response_analysis_params)

        flash_response_df = get_p_values_from_shuffled_spontaneous(session,flash_response_df)
        flash_response_df = add_image_name(session,flash_response_df)
        flash_response_df = annotate_flash_response_df_with_pref_stim(flash_response_df)

        output_fn = os.path.join(output_path, 'flash_response_df_{}.h5'.format(experiment_id))
        print('Writing flash response df to {}'.format(output_fn))
        flash_response_df.to_hdf(output_fn, key='df', complib='bzip2', complevel=9)

    elif case==1:

        nwb_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/nwb_files/behavior_ophys_session_880961028.nwb'
        api = BehaviorOphysNwbApi(nwb_path, filter_invalid_rois=True)
        session = BehaviorOphysSession(api)

        #Small data for testing
        session.__dict__['dff_traces'].value = session.dff_traces.iloc[:5]
        session.__dict__['stimulus_presentations'].value = session.stimulus_presentations.iloc[:20]

        response_analysis_params = {'window_around_timepoint_seconds':[-.5,.75], # -500ms, 750ms
                                   'response_window_duration_seconds': 0.75, 
                                   'baseline_window_duration_seconds': 0.5} 

        flash_response_df = get_flash_response_df(session, response_analysis_params)
        flash_response_df = get_p_values_from_shuffled_spontaneous(session,flash_response_df)
        flash_response_df = add_image_name(session,flash_response_df)
        flash_response_df = annotate_flash_response_df_with_pref_stim(flash_response_df)
