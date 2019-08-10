import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    for f in formats:
        fig.savefig(os.path.join(fig_dir, fig_title + f), transparent=True, orientation='landscape')


def get_dff_matrix(session):
    '''
        Returns the dff_trace of a session as a numpy matrix
    '''
    dff = np.stack(session.dff_traces.dff, axis=0)
    return dff


def get_mean_df(response_df, conditions=['cell_specimen_id', 'image_name'], flashes=False, omitted=False):
    '''
        Computes an analysis on a selection of responses (either flashes or trials). Computes mean_response, sem_response, the pref_stim, fraction_active_responses.

        INPUTS
        response_df, the dataframe to group
        conditions, the conditions to group by, the first entry should be 'cell_specimen_id', the second could be 'image_name' or 'change_image_name'
        flashes, if True, computes the fraction of individual images that were significant

        Returns a dataframe, does not alter the response_df
    '''

    rdf = response_df.copy()
    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace', 'mean_responses']]
    mdf = mdf.reset_index()

    if ('image_name' in conditions) or ('change_image_name' in conditions):
        mdf = annotate_mean_df_with_pref_stim(mdf)

    # What fraction of individual responses were significant?
    if flashes:
        fraction_significant_responses = rdf.groupby(conditions).apply(get_fraction_significant_responses)
        fraction_significant_responses = fraction_significant_responses.reset_index()
        mdf['fraction_significant_responses'] = fraction_significant_responses.fraction_significant_responses


    return mdf


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


def annotate_mean_df_with_pref_stim(mean_df):
    '''
        Calculates the preferred stimulus based on the mean_response index.
        Inputs: mean_df is a dataframe of the mean responses.
    '''
    if 'image_name' in mean_df.keys():
        image_name = 'image_name'
    else:
        image_name = 'change_image_name'
    mdf = mean_df.reset_index()
    mdf['pref_stim'] = False
    if 'cell_specimen_id' in mdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    for cell in mdf[cell_key].unique():
        mc = mdf[(mdf[cell_key] == cell)]
        mc = mc[mc[image_name] != 'omitted']
        temp = mc[(mc.mean_response == np.max(mc.mean_response.values))][image_name].values
        if len(temp) > 0:
            pref_image = temp[0]
            row = mdf[(mdf[cell_key] == cell) & (mdf[image_name] == pref_image)].index
            mdf.loc[row, 'pref_stim'] = True
    return mdf


def get_fraction_significant_responses(group, threshold=0.05):
    '''
        Calculates the fraction of trials or flashes that have a p_value below threshold
        We really need to think about multiple comparisons corrections here!
    '''
    fraction_significant_responses = len(group[group.p_value < threshold]) / float(len(group))
    return pd.Series({'fraction_significant_responses': fraction_significant_responses})


def get_xticks_xticklabels(trace, ophys_frame_rate=31., interval_sec=1, window=[-4, 8]):
    """
    Function that accepts a timeseries, evaluates the number of points in the trace,
    and converts from acquisition frames to timestamps relative to a given window of time covered by the trace.

    :param trace: a single trace where length = the number of timepoints
    :param ophys_frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param interval_sec: interval in seconds in between labels

    :return: xticks, xticklabels = xticks in units of ophys frames frames, xticklabels in seconds relative
    """
    interval_frames = interval_sec * ophys_frame_rate
    n_frames = len(trace)
    n_sec = n_frames / ophys_frame_rate
    xticks = np.arange(0, n_frames + 1, interval_frames)
    xticklabels = np.arange(0, n_sec + 0.1, interval_sec)
    if not window:
        xticklabels = xticklabels - n_sec / 2
    else:
        xticklabels = xticklabels + window[0]
    if interval_sec >= 1:
        xticklabels = [int(x) for x in xticklabels]
    return xticks, xticklabels


def plot_mean_trace(traces, window=[-4,8], interval_sec=1, ylabel='dF/F', legend_label=None, color='k', ax=None):
    """
    Function that accepts an array of single trial traces and plots the mean and SEM of the trace, with xticklabels in seconds

    :param traces: array of individual trial traces to average and plot. traces must be of equal length
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param y_label: 'dF/F' for calcium trace, 'running speed (cm/s)' for running speed trace
    :param legend_label: string describing trace for legend (ex: 'go', 'catch', image name or other condition identifier)
    :param color: color to plot the trace
    :param interval_sec: interval in seconds for x_axis labels
    :param xlims: range in seconds to plot. Must be <= the length of the traces
    :param ax: if None, create figure and axes to plot. If axis handle is provided, plot is created on that axis

    :return: axis handle
    """
    ophys_frame_rate = 31.
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.mean(traces, axis=0)
        times = np.arange(0, len(trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        ax.plot(trace, label=legend_label, linewidth=3, color=color)
        ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)

        xticks, xticklabels = get_xticks_xticklabels(trace, ophys_frame_rate, interval_sec, window=window)
        ax.set_xticks(xticks)
        if interval_sec < 1:
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(0, len(trace))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_flashes_on_trace(ax, window=[-4,8], go_trials_only=False, omitted=False, flashes=False, alpha=0.25, facecolor='gray'):
    """
    Function to create transparent gray bars spanning the duration of visual stimulus presentations to overlay on existing figure

    :param ax: axis on which to plot stimulus presentation times
    :param window: window of time the trace covers, in seconds
    :param trial_type: 'go' or 'catch'. If 'go', different alpha levels are used for stimulus presentations before and after change time
    :param omitted: boolean, use True if plotting response to omitted flashes
    :param alpha: value between 0-1 to set transparency level of gray bars demarcating stimulus times

    :return: axis handle
    """

    frame_rate = 31.
    stim_duration = .25
    blank_duration = .5
    change_frame = np.abs(window[0]) * frame_rate
    end_frame = (window[1] + np.abs(window[0])) * frame_rate
    interval = blank_duration + stim_duration
    if omitted:
        array = np.arange((change_frame + interval), end_frame, interval * frame_rate)
        array = array[1:]
    else:
        array = np.arange(change_frame, end_frame, interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if go_trials_only:
        alpha = alpha * 3
    else:
        alpha
    array = np.arange(change_frame - ((blank_duration) * frame_rate), 0, -interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] - (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax






def create_multi_session_mean_df(cache, experiment_ids, conditions=['cell_specimen_id','change_image_name'], flashes=False):
    '''
        Creates a mean response dataframe by combining multiple sessions. 
        
        manifest, the cache manifest
        Sessions, is a list of session objects to merge
        conditions is the set of conditions to send to get_mean_df() to merge. The first entry should be 'cell_specimen_id'
        flashes, if TRUE, merges the flash_response_df, otherwise merges the trial_response_df

        Returns a dataframe with index given by the session experiment ids. This allows for easy analysis like:
        mega_mdf.groupby('experiment_id').mean_response.mean()
    '''
    manifest = cache.manifest
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        session = cache.get_session(experiment_id)
        print(session.metadata['ophys_experiment_id'])
        if flashes:
            mdf = get_mean_df(session.flash_response_df,conditions=conditions)
        else:
            mdf = get_mean_df(session.trial_response_df,conditions=conditions)
        mdf['experiment_id'] = session.metadata['ophys_experiment_id']
        mdf['experiment_container_id'] = session.metadata['experiment_container_id']
        stage = manifest[manifest.ophys_experiment_id == session.metadata['ophys_experiment_id']].stage_name.values[0]
        mdf['stage_name']= stage
        mdf['passive'] = parse_stage_for_passive(stage)
        mdf['image_set'] = parse_stage_for_image_set(stage)
        mdf['targeted_structure'] = session.metadata['targeted_structure']
        mdf['imaging_depth'] = session.metadata['imaging_depth']
        mdf['full_genotype'] = session.metadata['full_genotype']
        mdf['cre_line'] = session.metadata['full_genotype'].split('/')[0]
        mdf['retake_number'] = manifest[manifest.ophys_experiment_id == session.metadata['ophys_experiment_id']].retake_number.values[0]

        mega_mdf = pd.concat([mega_mdf, mdf])
    mega_mdf = mega_mdf.reset_index()
    mega_mdf = mega_mdf.set_index('experiment_id')
    mega_mdf = mega_mdf.drop(columns=['level_0','index'])
    return mega_mdf

def parse_stage_for_passive(stage):
    return 'passive' in stage

def parse_stage_for_image_set(stage):
    return stage[15]





