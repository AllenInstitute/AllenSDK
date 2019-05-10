import numpy as np
import pandas as pd
import math
from typing import NamedTuple
import os

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import equals
from allensdk.deprecated import legacy


class BehaviorOphysSession(LazyPropertyMixin):
    """Represents data from a single Visual Behavior Ophys imaging session.  LazyProperty attributes access the data only on the first demand, and then memoize the result for reuse.
    
    Attributes:
        ophys_experiment_id : int (LazyProperty)
            Unique identifier for this experimental session
        segmentation_mask_image : SimpleITK.Image (LazyProperty)
            2D image of the segmented regions-of-interest in the field of view
        stimulus_timestamps : numpy.ndarray (LazyProperty)
            Timestamps associated the stimulus presentations on the monitor 
        ophys_timestamps : numpy.ndarray (LazyProperty)
            Timestamps associated with frames captured by the microscope
        metadata : dict (LazyProperty)
            A dictionary of session-specific metadata
        dff_traces : pandas.DataFrame (LazyProperty)
            The traces of dff organized into a dataframe; index is the cell roi ids
        cell_specimen_table : pandas.DataFrame (LazyProperty)
            Cell roi information organized into a dataframe; index is the cell roi ids
        running_speed : allensdk.brain_observatory.running_speed.RunningSpeed (LazyProperty)
            NamedTuple with two fields
                timestamps : numpy.ndarray
                    Timestamps of running speed data samples
                values : np.ndarray
                    Running speed of the experimental subject (in cm / s).
        running_data_df : pandas.DataFrame (LazyProperty)
            Dataframe containing various signals used to compute running speed
        stimulus_presentations : pandas.DataFrame (LazyProperty)
            Table whose rows are stimulus presentations (i.e. a given image, for a given duration, typically 250 ms) and whose columns are presentation characteristics.
        stimulus_templates : dict (LazyProperty)
            A dictionary containing the stimulus images presented during the session keys are data set names, and values are 3D numpy arrays.
        licks : pandas.DataFrame (LazyProperty)
            A dataframe containing lick timestamps
        rewards : pandas.DataFrame (LazyProperty)
            A dataframe containing timestamps of delivered rewards
        task_parameters : dict (LazyProperty)
            A dictionary containing parameters used to define the task runtime behavior
        trials : pandas.DataFrame (LazyProperty)
            A dataframe containing behavioral trial start/stop times, and trial data
        corrected_fluorescence_traces : pandas.DataFrame (LazyProperty)
            The motion-corrected fluorescence traces organized into a dataframe; index is the cell roi ids
        average_image : SimpleITK.Image (LazyProperty)
            2D image of the microscope field of view, averaged across the experiment
        motion_correction : pandas.DataFrame LazyProperty
            A dataframe containing trace data used during motion correction computation
    """

    @classmethod
    def from_LIMS(cls, ophys_experiment_id):
        return cls(api=BehaviorOphysLimsApi(ophys_experiment_id))

    def __init__(self, api=None):

        self.api = api

        self.ophys_experiment_id = LazyProperty(self.api.get_ophys_experiment_id)
        self.segmentation_mask_image = LazyProperty(self.api.get_segmentation_mask_image)
        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps)
        self.metadata = LazyProperty(self.api.get_metadata)
        self.dff_traces = LazyProperty(self.api.get_dff_traces)
        self.cell_specimen_table = LazyProperty(self.api.get_cell_specimen_table)
        self.running_speed = LazyProperty(self.api.get_running_speed)
        self.running_data_df = LazyProperty(self.api.get_running_data_df)
        self.stimulus_presentations = LazyProperty(self.api.get_stimulus_presentations)
        self.stimulus_templates = LazyProperty(self.api.get_stimulus_templates)
        self.licks = LazyProperty(self.api.get_licks)
        self.rewards = LazyProperty(self.api.get_rewards)
        self.task_parameters = LazyProperty(self.api.get_task_parameters)
        self.trials = LazyProperty(self.api.get_trials)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces)
        self.average_image = LazyProperty(self.api.get_average_image)
        self.motion_correction = LazyProperty(self.api.get_motion_correction)

    @legacy('Consider using "get_dff_timeseries" instead.')
    def get_dff_traces(self, cell_specimen_ids=None):

        if cell_specimen_ids is None:
            cell_specimen_ids = self.get_cell_specimen_ids()

        csid_table = self.cell_specimen_table[['cell_specimen_id']]
        csid_subtable = csid_table[csid_table['cell_specimen_id'].isin(cell_specimen_ids)]
        dff_table = csid_subtable.join(self.dff_traces, how='left')
        dff_traces = np.vstack(dff_table['dff'].values)
        timestamps = self.ophys_timestamps

        assert (len(cell_specimen_ids), len(timestamps)) == dff_traces.shape
        return timestamps, dff_traces

    @legacy()
    def get_cell_specimen_indices(self, cell_specimen_ids):
        csid_table = self.cell_specimen_table[['cell_specimen_id']].set_index('cell_specimen_id')
        return [csid_table.index.get_loc(csid) for csid in cell_specimen_ids]

    @legacy('Consider using "cell_specimen_table[\'cell_specimen_id\']" instead.')
    def get_cell_specimen_ids(self):
        return self.cell_specimen_table['cell_specimen_id'].values


if __name__ == "__main__":

    # from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi

    # blacklist = [797257159, 796306435, 791453299, 809191721, 796308505, 798404219] #
    # api_list = []
    # df = BehaviorOphysLimsApi.get_ophys_experiment_df()
    # for cid in [791352433, 814796698, 814796612, 814796558, 814797528]:
    #     df2 = df[(df['container_id'] == cid) & (df['workflow_state'] == 'passed')]
    #     api_list += [BehaviorOphysLimsApi(oeid) for oeid in df2['ophys_experiment_id'].values if oeid not in blacklist]

    # for api in api_list:

    #     session = BehaviorOphysSession(api=api)
    #     if len(session.licks) > 100:

    #         print(api.get_ophys_experiment_id())

        # session.segmentation_mask_image
        # session.stimulus_timestamps
        # session.ophys_timestamps
        # session.metadata
        # session.dff_traces
        # session.cell_specimen_table
        # session.running_speed
        # session.running_data_df

        # print(api.get_ophys_experiment_id(), len(session.licks), session.metadata['experiment_datetime'])
        # session.rewards
        # session.task_parameters
        # session.trials
        # session.corrected_fluorescence_traces
        # session.average_image
        # session.motion_correction

            # nwb_filepath = '/allen/aibs/technology/nicholasc/tmp/behavior_ophys_session_{get_ophys_experiment_id}.nwb'.format(get_ophys_experiment_id=api.get_ophys_experiment_id())
            # BehaviorOphysNwbApi(nwb_filepath).save(session)
            # assert equals(session, BehaviorOphysSession(api=BehaviorOphysNwbApi(nwb_filepath)))




        # print(session.running_speed)


    # nwb_filepath = '/home/nicholasc/projects/allensdk/tmp.nwb'
    # session = BehaviorOphysSession(789359614)
    # nwb_api = BehaviorOphysNwbApi(nwb_filepath)
    # nwb_api.save(session)

    # print(session.cell_specimen_table)


    # api_2 = BehaviorOphysNwbApi(nwb_filepath)
    # session2 = BehaviorOphysSession(789359614, api=api_2)
    
    # assert session == session2

    session = BehaviorOphysSession.from_LIMS(789359614)
    # session.segmentation_mask_image
    # session.stimulus_timestamps
    # session.ophys_timestamps
    # session.metadata
    # session.dff_traces
    # session.cell_specimen_table
    # running_speed
    # print(session.stimulus_index)
    # session.running_data_df
    print(session.stimulus_presentations)
    # session.stimulus_templates
    # session.stimulus_index
    # session.licks
    # session.rewards
    # session.task_parameters
    # session.trials
    # session.corrected_fluorescence_traces
    # session.average_image
    # session.motion_correction
