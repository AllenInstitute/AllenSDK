from typing import Optional, List

import pandas as pd
import numpy as np
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_presentations, get_stimulus_metadata, is_change_event
from allensdk.brain_observatory.nwb import \
    create_stimulus_presentation_time_interval, get_column_name
from allensdk.brain_observatory.nwb.nwb_api import NwbApi


def stim_name_parse(stim_name):

    stim_name = stim_name[:-4]
    components = stim_name.split('_')

    session_number = int(components[-2])
    segment_number = int(components[-1])

    if components[-3]=='test':
        test_or_train = 'test'
    else:
        test_or_train = 'train'

    return session_number, segment_number, test_or_train

def get_original_stim_name(stage_number, segment_number, test_or_train):
    
    if test_or_train=='test':
        original_stim_name = 'Session_test_'+str(stage_number)+'_'+str(segment_number)+'.npy'
    if test_or_train=='train':
        original_stim_name = 'Session_'+str(stage_number)+'_'+str(segment_number)+'.npy'

    return original_stim_name

def shorten_stimulus_presentation(msn_stim_table):

    min_table=msn_stim_table.groupby(['stimulus_template', 'trial_number'],as_index=False).min()
    max_table=msn_stim_table.groupby(['stimulus_template', 'trial_number'],as_index=False).max()

    min_table['end_frame'] = max_table['end_frame']
    min_table['end_time'] = max_table['end_time']
    min_table['duration'] = min_table['end_time']-min_table['start_time']
    min_table['stimulus_start_index'] = min_table['stimulus_index']
    min_table['stimulus_length'] = max_table['stimulus_index']+1

    short_table = min_table
    short_table = short_table.drop(columns=['stimulus_index'])
    
    return short_table.sort_values(by='start_frame').reset_index(drop=True)


class DenseMoviePresentations(DataObject, StimulusFileReadableInterface,
                    NwbReadableInterface, NwbWritableInterface):
    """Stimulus presentations"""
    def __init__(self, presentations: pd.DataFrame):
        super().__init__(name='presentations', value=presentations)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        raise NotImplementedError
        return None

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "DenseMoviePresentations":
        raise NotImplementedError

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps) -> "DenseMoviePresentations":
        """Get stimulus presentation data.

        :param stimulus_file
        :param stimulus_timestamps


        :returns: pd.DataFrame --
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration)
            and whose columns are presentation characteristics.
        """
        timestamps = stimulus_timestamps.value
        pkl_data = stimulus_file.data

        stimulus_presentation_table = pd.DataFrame()

        pre_blank = int(pkl_data['pre_blank_sec']*pkl_data['fps'])

        for stim in pkl_data['stimuli']:

            warped_stim_name = str(stim['movie_path']).split('\\')[-1]

            stage_number, segment_number, test_or_train = stim_name_parse(warped_stim_name)
            original_stim_name = get_original_stim_name(stage_number, segment_number, test_or_train)

            frame_list = np.array(stim['frame_list'])

            frame_index = frame_list[frame_list!=-1][::2]
            indices = np.where(frame_list!=-1)[0]
            start_frames = indices[::2] + pre_blank
            end_frames = start_frames + 2
            start_times = timestamps[start_frames]
            end_times = timestamps[end_frames]
            duration = end_times - start_times
            trial_number = np.arange(frame_index.shape[0])//(np.max(frame_index)+1)

            data = np.vstack([frame_index, start_frames, end_frames, start_times, end_times, duration, trial_number]).T
            temp_df = pd.DataFrame(data, columns=('stimulus_index', 'start_frame', 'end_frame', 'start_time', 'end_time', 'duration', 'trial_number'))

            temp_df['warped_stimulus'] = warped_stim_name
            temp_df['stimulus_template'] = original_stim_name
            temp_df['stage'] = stage_number
            temp_df['segment'] = segment_number
            temp_df['test_or_train'] = test_or_train

            stimulus_presentation_table = stimulus_presentation_table.append(temp_df, ignore_index=True)

        stimulus_presentation_table = stimulus_presentation_table.sort_values(by='start_frame').reset_index(drop=True)

        return DenseMoviePresentations(presentations=stimulus_presentation_table)

        

