import numpy as np
import pandas as pd
import math
from typing import NamedTuple
import os

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import equals


class BehaviorOphysSession(LazyPropertyMixin):

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
    df = session.stimulus_presentations
    # df2 = session.stimulus_index

    print(df.head())
    # print(df2.index.values)

    # print(df.merge(df2, left_on='start_time', right_index=True))


    # session.segmentation_mask_image
    # session.stimulus_timestamps
    # session.ophys_timestamps
    # session.metadata
    # session.dff_traces
    # session.cell_specimen_table
    # session.running_speed
    # session.running_data_df
    # session.stimulus_presentations
    # session.stimulus_templates
    # session.stimulus_index
    # session.licks
    # session.rewards
    # session.task_parameters
    # session.trials
    # session.corrected_fluorescence_traces
    # session.average_image
    # session.motion_correction
