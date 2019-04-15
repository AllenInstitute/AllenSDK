# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:11:23 2016

@author: Xiaoxuan Jia
"""

import ast
import re

import numpy as np
import pandas as pd

import warnings


REPR_PARAMS_RE = re.compile(r'([a-z0-9]+=[^=]+)[,\)]', re.IGNORECASE)
ARRAY_RE = re.compile(r'array\((?P<contents>\[.*\])\)')


def create_stim_table(
    stimuli, stimulus_tabler, spontaneous_activity_tabler, 
    sort_key='Start', block_key='stimulus_block', index_key='stimulus_index'
):
    ''' Build a full stimulus table

    Parameters
    ----------
    stimuli : list of dict
        Each element is a stimulus dictionary, as provided by the stim.pkl file.
    stimulus_tabler : function 
        A function which takes a single stimulus dictionary as its argument and returns a stimulus table dataframe.
    spontaneous_activity_tabler : function
        A function which takes a list of stimulus tables as arguments and returns a list of 0 or more tables 
        describing spontaneous activity sweeps.
    sort_key : str, optional
        Sort the final stimulus table in ascending order by this key. Defaults to 'Start'.

    Returns
    -------
    stim_table_full : pandas.DataFrame
        Each row is a sweep. Has columns describing (in frames) the start and end times of each sweep. Other columns 
        describe the values of stimulus parameters on those sweeps.

    '''
    
    stimulus_tables = []
    max_index = 0

    for ii, stimulus in enumerate(stimuli):
        current_tables = stimulus_tabler(stimulus)
        for table in current_tables:
            table[index_key] = ii

        stimulus_tables.extend(current_tables)
                    
    stimulus_tables = sorted(stimulus_tables, key=lambda df: min(df[sort_key].values))
    for ii, stim_table in enumerate(stimulus_tables):
        stim_table[block_key] = ii

    stimulus_tables.extend(spontaneous_activity_tabler(stimulus_tables))

    stim_table_full = pd.concat(stimulus_tables, ignore_index=True, sort=False)
    stim_table_full.sort_values(by=[sort_key], inplace=True)
    stim_table_full.reset_index(drop=True, inplace=True)          

    return stim_table_full


def make_spontaneous_activity_tables(stimulus_tables, start_key='Start', end_key='End', duration_threshold=0.0):
    ''' Fills in frame gaps in a set of stimulus tables. Suitable for use as the spontaneous_activity_tabler in 
    create_stim_table.

    Parameters
    ----------
    stimulus_tables : list of pd.DataFrame
        Input tables - should have start_key and end_key columns.
    start_key : str, optional
        Column name for the start of a sweep. Defaults to 'Start'.
    end_key : str, optional
        Column name for the end of a sweep. Defaults to 'End'.
    duration_threshold : numeric or None
        If not None (default is 0), remove spontaneous activity sweeps whose duration is 
        less than this threshold. 

    Returns
    -------
    list : 
        Either empty, or contains a single pd.DataFrame. The rows of the dataframe are spontenous activity sweeps.

    '''

    nstimuli = len(stimulus_tables)
    if nstimuli == 0:
        return []

    spon_start = np.zeros(nstimuli + 1, dtype=int)
    spon_end = np.zeros(nstimuli, dtype=int)

    for ii, table in enumerate(stimulus_tables):
        spon_start[ii + 1] = table[end_key].values[-1]
        spon_end[ii] = table[start_key].values[0]

    spon_start = spon_start[:-1]
    spon_sweeps = pd.DataFrame({start_key: spon_start, end_key: spon_end})

    if duration_threshold is not None:
        spon_sweeps = spon_sweeps[np.fabs(spon_sweeps[start_key] - spon_sweeps[end_key]) > duration_threshold]
        spon_sweeps.reset_index(drop=True, inplace=True)

    return [spon_sweeps]


def apply_frame_times(stimulus_table, frame_times, frames_per_second=None, extra_frame_time=False, map_columns=('Start', 'End')):
    ''' Converts sweep times from frames to seconds.

    Parameters
    ----------
    stimulus_table : pd.DataFrame
        Rows are sweeps. Columns are stimulus parameters as well as start and end frames for each sweep.
    frame_times : numpy.ndarrray
        Gives the time in seconds at which each frame (indices) began.
    frames_per_second : numeric, optional
        If provided, and extra_frame_time is True, will be used to calculcate the extra_frame_time.
    extra_frame_time : float, optional
        If provided, an additional frame time will be appended. The time will be incremented by extra_frame_time from 
        the previous last frame time, to denote the time at which the last frame ended. If False, no extra time will be 
        appended. If None (default), the increment will be 1.0/fps.
    map_columns : tuple of str, optional
        Which columns to replace with times. Defaults to 'Start' and 'End

    Returns
    -------
    stimulus_table : pd.DataFrame
        As above, but with map_columns values converted to seconds from frames.

    '''

    stimulus_table = stimulus_table.copy()

    if extra_frame_time is True and frames_per_second is not None:
        extra_frame_time = 1.0 / frames_per_second
    if extra_frame_time is not False:
        frame_times = np.append(frame_times, frame_times[-1] + extra_frame_time )

    for column in map_columns:
        stimulus_table[column] = frame_times[np.around(stimulus_table[column]).astype(int)]

    return stimulus_table


def apply_display_sequence(
    sweep_frames_table, frame_display_sequence, 
    start_key='Start', end_key='End', diff_key='dif', block_key='stimulus_block'
):
    ''' Adjust raw sweep frames for a stimulus based on the display sequence 
    for that stimulus.

    Parameters
    ----------
    sweep_frames_table : pd.DataFrame
        Each row is a sweep. Has two columns, 'start' and 'end', 
        which describe (in frames) when that sweep began and ended.
    frame_display_sequence : np.ndarray
        2D array. Rows are display intervals. The 0th column is the start frame of 
        that interval, the 1st the end frame.

    Returns
    -------
    sweep_frames_table : pd.DataFrame
        As above, but start and end frames have been adjusted based on the display sequence.

    Notes
    -----
    The frame values in the raw sweep_frames_table are given in 0-indexed offsets from the 
    start of display for this stimulus. This domain only takes into account frames which are part
    of a display interval for that stimulus, so the frame ids need to be adjusted to lie on the global 
    frame sequence.

    '''

    sweep_frames_table = sweep_frames_table.copy()
    if not block_key in sweep_frames_table.columns.values:
        sweep_frames_table[block_key] = np.zeros((sweep_frames_table.shape[0]), dtype=int)

    sweep_frames_table[diff_key] = sweep_frames_table[end_key] - sweep_frames_table[start_key]

    sweep_frames_table[start_key] += frame_display_sequence[0, 0]
    for seg in range(len(frame_display_sequence) - 1):
        match_inds = sweep_frames_table[start_key] >= frame_display_sequence[seg, 1]

        sweep_frames_table.loc[match_inds, start_key] += frame_display_sequence[seg + 1, 0] - frame_display_sequence[seg, 1]
        sweep_frames_table.loc[match_inds, block_key] = seg + 1

    sweep_frames_table[end_key] = sweep_frames_table[start_key] + sweep_frames_table[diff_key]
    sweep_frames_table = sweep_frames_table[sweep_frames_table[end_key] <= frame_display_sequence[-1, 1]]
    sweep_frames_table = sweep_frames_table[sweep_frames_table[start_key] <= frame_display_sequence[-1, 1]]    

    sweep_frames_table.drop(diff_key, inplace=True, axis=1)
    return sweep_frames_table


def read_stimulus_name_from_path(stimulus):
    '''Obtains a human-readable stimulus name by looking at the filename of the 'stim_path' item.

    Parameters
    ----------
    stimulus : dict
        must contain a 'stim_path' item.
    
    Returns
    -------
    str : 
        name of stimulus

    '''

    return stimulus['stim_path'].split('\\')[-1].split('.')[0]


def extract_const_params_from_stim_repr(stim_repr, repr_params_re=REPR_PARAMS_RE, array_re=ARRAY_RE):
    '''Parameters which are not set as sweep_params in the stimulus script (usually because they are not 
    varied during the course of the session) are not output in an easily machine-readable format. This function 
    attempts to recover them by parsing the string repr of the stimulus.

    Parameters
    ----------
        stim_repr : str
            The repr of the camstim stimulus object. Served up per-stimulus in the stim pickle.
        repr_params_re : re.Pattern
            Extracts attributes as "="-seperated strings
        array_re : re.Pattern
            Extracts list reprs from numpy array reprs.

    Returns
    -------
    repr_params : dict
        dictionary of paramater keys and values extracted from the stim repr. Where possible, the values are converted 
        to native Python types.

    '''

    repr_params = {}

    for match in repr_params_re.findall(stim_repr):
        k, v = match.split('=')
        
        if k not in repr_params:

            m = array_re.match(v)
            if m is not None:
                v = m['contents']

            try:
                v = ast.literal_eval(v)
            except ValueError as err:
                pass

            repr_params[k] = v

        else:
            raise KeyError(f'duplicate key: {k}')

    return repr_params


def build_stimuluswise_table(
    stimulus, seconds_to_frames, 
    start_key='Start', end_key='End', name_key='stimulus_name', block_key='stimulus_block',
    get_stimulus_name=None
):
    ''' Construct a table of sweeps, including their times on the experiment-global clock 
    and the values of each relevant parameter.

    Parameters
    ----------
    stimulus : dict
        Describes presentation of a stimulus on a particular experiment. Has a number of fields, 
        of which we are using:
            stim_path : str
                windows file path to the stimulus data
            sweep_frames : list of lists
                rows are sweeps, columns are start and end frames of that sweep 
                (in the stimulus-specific frame domain). C-order.
            sweep_order : list of int
                indices are frames, values are the sweep on that frame
            display_sequence : list of list
                rows are intervals in which the stimulus was displayed. Columns are start 
                and end times (s, global) of the display. C-order.
             dimnames : list of str
                Names of parameters for this stimulus (such as "Contrast")
            sweep_table : list of tuple
                Each element is a tuple of parameter values (1 per dimname) describing 
                a single sweep.
    seconds_to_frames : function
        Converts experiment seconds to frames
    start_key : str, optional
        key to use for start frame indices. Defaults to 'Start'
    end_key : str, optional
        key to use for end frame indices. Defaults to 'End'
    name_key : str, optional
        key to use for stimulus name annotations. Defaults to 'stimulus_name'
    block_key : str, optional
        key to use for the 0-index position of this stimulus block
    get_stimulus_name : function | dict -> str, optional
        extracts stimulus name from the stimulus dictionary. Default is read_stimulus_name_from_path

    Returns
    -------
    list of pandas.DataFrame :
        Each table corresponds to an entry in the display sequence. 
        Rows are sweeps, columns are stimulus parameter values as well as "Start" and "End".

    '''

    if get_stimulus_name is None:
        get_stimulus_name = read_stimulus_name_from_path

    frame_display_sequence = seconds_to_frames(stimulus['display_sequence'])

    sweep_frames_table = pd.DataFrame(stimulus['sweep_frames'], columns=(start_key, end_key))    
    sweep_frames_table[block_key] = np.zeros([sweep_frames_table.shape[0]], dtype=int)
    sweep_frames_table = apply_display_sequence(sweep_frames_table, frame_display_sequence, block_key=block_key)

    stim_table = pd.DataFrame({
        start_key: sweep_frames_table[start_key], 
        end_key: sweep_frames_table[end_key] + 1,
        name_key: get_stimulus_name(stimulus),
        block_key: sweep_frames_table[block_key]
    })

    sweep_order = stimulus['sweep_order'][:len(sweep_frames_table)]
    dimnames = stimulus['dimnames']
    
    if not dimnames or 'ReplaceImage' in dimnames:
        stim_table['Image'] = sweep_order
    else:
        stim_table['sweep_number'] = sweep_order
        sweep_table = pd.DataFrame(stimulus['sweep_table'], columns=dimnames)
        sweep_table['sweep_number'] = sweep_table.index

        stim_table = assign_sweep_values(stim_table, sweep_table)
        stim_table = split_column(stim_table, 'Pos', {'Pos_x': lambda field: field[0], 'Pos_y': lambda field: field[1]})
   
    const_params = parse_stim_repr(stimulus['stim'])
    existing_columns = set(stim_table.columns)
    for const_param_key, const_param_value in const_params.items():
        if const_param_key not in existing_columns:
            stim_table[const_param_key] = [const_param_value] * stim_table.shape[0]
        else:
            warnings.warn(f'found sweep_param named: {const_param_key}, ignoring const param of the same name (value: {const_param_value})')

    unique_indices = np.unique(stim_table[block_key].values)
    output = [ stim_table.loc[stim_table[block_key] == ii, :] for ii in unique_indices ]

    return output


def split_column(table, column, new_columns, drop_old=True):
    ''' Divides a dataframe column into multiple columns.

    Parameters
    ----------
    table : pandas.DataFrame
        Columns will be drawn from and assigned to this dataframe. This dataframe will NOT be modified inplace.
    column : str
        This column will be split.
    new_columns : dict, mapping strings to functions
        Each key will be the name of a new column, while its value (a function) will be used to build the 
        new column's values. The functions should map from a single value of the original column to a single value 
        of the new column. 
    drop_old : bool, optional
        If True, the original column will be dropped from the table.

    Returns
    -------
    table : pd.DataFrame
        The modified table

    '''

    if not column in table:
        return table
    table = table.copy()

    for new_column, rule in new_columns.items():
        table[new_column] = table[column].apply(rule)
    
    if drop_old:
        table.drop(column, inplace=True, axis=1) 
    return table


def assign_sweep_values(stim_table, sweep_table, on='sweep_number', drop=True, tmp_suffix='_stimtable_todrop'):
    ''' Left joins a stimulus table to a sweep table in order to associate epochs in time with stimulus characteristics.
    
    Parameters
    ----------
    stim_table : pd.DataFrame
        Each row is a stimulus epoch, with start and end times and a foreign key onto a particular sweep.
    sweep_table : pd.DataFrame
        Each row is a sweep. Should have columns in common with the stim_table - the resulting table will use values from 
        the sweep_table.
    on : str, optional
        Column on which to join.
    drop : bool, optional
        If True (default), the join column (argument on) will be dropped from the output.
    tmp_suffix : str, optional
        Will be used to identify overlapping columns. Should not appear in the name of any column in either dataframe.

    '''

    joined_table = stim_table.join(sweep_table, on=on, lsuffix=tmp_suffix)
    for dim in joined_table.columns.values:
        if tmp_suffix in dim:
            joined_table.drop(dim, inplace=True, axis=1)

    if drop:
        joined_table.drop(on, inplace=True, axis=1)  
    return joined_table


def create_stim_table_ephys_probe(data, events, probe_start_index, led_rise=None): # pragma: no cover
    '''

    Parameters
    ----------
    data : Object
        Experiment stimulus record (historically, stimulus pkl).
    events : dict
        As read from openephys loadEvents
    probe_start_index : int
    led_rise : np.ndarray, optional
        led times (as extracted from the sync dataset). TODO: remove as is not used

    Notes
    -----
    This function is deprecated!

    TODO : replace placeholder descriptions
    '''
    warnings.warn(np.VisibleDeprecationWarning(
        'This function is deprecated! Use create_stim_table to obtain master clock stimulus times, '
        'then use the align_timestamps module to align each probe\'s data'
    ))

    sync_frame_probe=events['timestamps'] / 30000.
    start_time = probe_start_index / 30000.

    ch0_events = np.where(events['channel'] == 3)
    on_events = np.where(events['eventId'] == 1)
    off_events = np.where(events['eventId'] == 0)
        
    on_times = np.intersect1d(ch0_events, on_events)
    off_times = np.intersect1d(ch0_events, off_events)
        
    T_on = events['timestamps'][on_times]/events['header']['sampleRate']
    T_off = events['timestamps'][off_times]/events['header']['sampleRate']

    d = np.diff(T_on)
    a = np.where(d > 0)[0]
    frame_start_times = T_on[a] 
    stim_frame_rise = frame_start_times-start_time
    
    #build stimulus tables for each stimulus type
    curr=0
    stimulus_spon_start = np.zeros(len(data['stimuli'])) #start frame of each epoch
    stimulus_spon_end = np.zeros(len(data['stimuli'])) #end frame of each epoch
    sync_table_full = pd.DataFrame()
    sync_table_dict = {}
    for curr in range(len(data['stimuli'])):
        display_sequence = np.array(data['stimuli'][curr]['display_sequence']).astype(float)   #in seconds
        display_sequence += data['pre_blank_sec']   #in seconds
        display_sequence *= data['fps']     #in stimulus frames

        sweep_frames = data['stimuli'][curr]['sweep_frames']
        stimulus_table = pd.DataFrame(sweep_frames,columns=('start','end'))            
        stimulus_table['dif'] = stimulus_table['end']-stimulus_table['start']
        stimulus_table.start += display_sequence[0,0]
        for seg in range(len(display_sequence)-1):
            for index, row in stimulus_table.iterrows():
                if row.start >= display_sequence[seg,1]:
                    stimulus_table.start[index] = stimulus_table.start[index] - display_sequence[seg,1] + display_sequence[seg+1,0]
        stimulus_table.end = stimulus_table.start+stimulus_table.dif
        print(len(stimulus_table))
        stimulus_table = stimulus_table[stimulus_table.end <= display_sequence[-1,1]]
        stimulus_table = stimulus_table[stimulus_table.start <= display_sequence[-1,1]]            
        print(len(stimulus_table))

        if curr <len(data['stimuli'])-1:
            stimulus_spon_start[curr+1] = int(stimulus_table['end'].values[-1])
        stimulus_spon_end[curr] = int(stimulus_table['start'].values[0])

        # frame number for end need to +1 to include the duration of that frame
        sync_table = pd.DataFrame(np.column_stack((stim_frame_rise[[int(i) for i in stimulus_table['start'].values]],stim_frame_rise[[int(i)+1 for i in stimulus_table['end'].values]])), columns=('Start', 'End'))

        sweep_order = data['stimuli'][curr]['sweep_order']
        sweep_order =  sweep_order[:len(stimulus_table)]
        dimnames = data['stimuli'][curr]['dimnames']
        stim_name = data['stimuli'][curr]['stim_path'].split('\\')[-1].split('.')[0]
        print(stim_name )
        
        if 'ReplaceImage' in dimnames:
            sync_table['Image'] = sweep_order[:len(stimulus_table)]
            curr += 1
        else:
            sync_table['sweep_number'] = sweep_order[:len(stimulus_table)]  #should be able to remove the len(stimulus_table) when stimulus is fixed appropriately
            sweep_table = data['stimuli'][curr]['sweep_table']
            sweeptable = pd.DataFrame(sweep_table, columns=dimnames)

            for dimname in dimnames:
                if dimname=='Pos':
                    sync_table[dimname+'_x'] = np.NaN
                    sync_table[dimname+'_y'] = np.NaN
                else:
                    sync_table[dimname] = np.NaN

            for index, row in sync_table.iterrows():
                if row['sweep_number'] >= 0:
                    for dimname in dimnames:
                        if dimname=='Pos':
                            sync_table[dimname+'_x'][index] = sweeptable[dimname][int(row['sweep_number'])][0]
                            sync_table[dimname+'_y'][index] = sweeptable[dimname][int(row['sweep_number'])][1]
                        else:
                            sync_table[dimname][index] = sweeptable[dimname][int(row['sweep_number'])]
                else:
                    for dimname in dimnames:
                        sync_table[dimname][index] = np.NaN
            sync_table.drop('sweep_number', inplace=True, axis=1)  
            curr += 1

        #add to the full sync table    
        sync_table_dict[stim_name] = sync_table
        sync_table_full = pd.concat([sync_table_full, sync_table], ignore_index=True)
        sync_table_full.sort(columns='Start', inplace=True, na_position='first')
        sync_table_full.reset_index(drop=True, inplace=True)
            
    #get spontaneous activity epochs
    stimulus_spon_start = np.array([int(i) for i in stimulus_spon_start])
    stimulus_spon_end = np.array([int(i) for i in stimulus_spon_end])
    spt_sync_table = pd.DataFrame(np.column_stack((stim_frame_rise[stimulus_spon_start],stim_frame_rise[stimulus_spon_end])), columns=('Start', 'End'))
    if len(stimulus_spon_start)>0:
        sync_table_dict['spontaneous'] = spt_sync_table
        sync_table_full = pd.concat([sync_table_full, spt_sync_table], ignore_index=True)

    sync_table_full.sort(columns='Start', inplace=True)#, na_position='first')    #want final dataframe to be consistent - eg. sorted by time
    sync_table_full.reset_index(drop=True, inplace=True)            
        
    return(sync_table_full,sync_table_dict, LED_rise)
