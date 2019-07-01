import numpy as np
import pandas as pd
import math
from typing import NamedTuple
import os

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import equals
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession

class BehaviorOphysSession(BehaviorSession, LazyPropertyMixin):
    """Represents data from a single Visual Behavior Ophys imaging session.  LazyProperty attributes access the data only on the first demand, and then memoize the result for reuse.
    
    Attributes:
        ophys_experiment_id : int (LazyProperty)
            Unique identifier for this experimental session
        max_projection : SimpleITK.Image (LazyProperty)
            2D max projection image
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
        average_projection : SimpleITK.Image (LazyProperty)
            2D image of the microscope field of view, averaged across the experiment
        motion_correction : pandas.DataFrame LazyProperty
            A dataframe containing trace data used during motion correction computation
    """

    @classmethod
    def from_lims(cls, ophys_experiment_id):
        return cls(api=BehaviorOphysLimsApi(ophys_experiment_id))

    def __init__(self, api=None):

        super(BehaviorOphysSession, self).__init__(api=api)

        self.ophys_experiment_id = LazyProperty(self.api.get_ophys_experiment_id)
        self.max_projection = LazyProperty(self.api.get_max_projection)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps)
        self.dff_traces = LazyProperty(self.api.get_dff_traces)
        self.cell_specimen_table = LazyProperty(self.api.get_cell_specimen_table)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces)
        self.average_projection = LazyProperty(self.api.get_average_projection)
        self.motion_correction = LazyProperty(self.api.get_motion_correction)
        self.segmentation_mask_image = LazyProperty(self.api.get_segmentation_mask_image)


    @legacy('Consider using "get_dff_timeseries" instead.')
    def get_dff_traces(self, cell_specimen_ids=None):

        if cell_specimen_ids is None:
            cell_specimen_ids = self.get_cell_specimen_ids()

        csid_table = self.cell_specimen_table.reset_index()[['cell_specimen_id']]
        csid_subtable = csid_table[csid_table['cell_specimen_id'].isin(cell_specimen_ids)].set_index('cell_specimen_id')
        dff_table = csid_subtable.join(self.dff_traces, how='left')
        dff_traces = np.vstack(dff_table['dff'].values)
        timestamps = self.ophys_timestamps

        assert (len(cell_specimen_ids), len(timestamps)) == dff_traces.shape
        return timestamps, dff_traces

    @legacy()
    def get_cell_specimen_indices(self, cell_specimen_ids):
        return [self.cell_specimen_table.index.get_loc(csid) for csid in cell_specimen_ids]

    @legacy('Consider using "cell_specimen_table[\'cell_specimen_id\']" instead.')
    def get_cell_specimen_ids(self):
        cell_specimen_ids = self.cell_specimen_table.index.values

        if np.isnan(cell_specimen_ids.astype(float)).sum() == len(self.cell_specimen_table):
            raise ValueError(f'cell_specimen_id values not assigned for {self.ophys_experiment_id}')
        return cell_specimen_ids


if __name__ == "__main__":

    ophys_experiment_id = 789359614
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)
    print(session.trials['reward_time'])
