import numpy as np
import pandas as pd
import math
from typing import NamedTuple

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi


class BehaviorOphysSession(LazyPropertyMixin):

    def __init__(self, ophys_experiment_id, api=None):

        self.ophys_experiment_id = ophys_experiment_id
        self.api = BehaviorOphysLimsApi(ophys_experiment_id) if api is None else api

        self.max_projection = LazyProperty(self.api.get_max_projection)
        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps)
        self.metadata = LazyProperty(self.api.get_metadata)
        self.dff_traces = LazyProperty(self.api.get_dff_traces)
        self.cell_specimen_table = LazyProperty(self.api.get_cell_specimen_table)
        self.running_speed = LazyProperty(self.api.get_running_speed)
        self.running_data_df = LazyProperty(self.api.get_running_data_df)
        self.stimulus_presentations = LazyProperty(self.api.get_stimulus_presentations)
        self.stimulus_templates = LazyProperty(self.api.get_stimulus_templates)
        self.stimulus_index = LazyProperty(self.api.get_stimulus_index)
        self.licks = LazyProperty(self.api.get_licks)
        self.rewards = LazyProperty(self.api.get_rewards)
        self.task_parameters = LazyProperty(self.api.get_task_parameters)
        self.trials = LazyProperty(self.api.get_trials)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces)
        self.average_image = LazyProperty(self.api.get_average_image)
        self.motion_correction = LazyProperty(self.api.get_motion_correction)


if __name__ == "__main__":

    # from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
    # nwb_filepath = '/home/nicholasc/projects/allensdk/tmp.nwb'
    # session = BehaviorOphysSession(789359614)
    # nwb_api = BehaviorOphysNwbApi(nwb_filepath)
    # nwb_api.save(session)

    # print(session.cell_specimen_table)


    # api_2 = BehaviorOphysNwbApi(nwb_filepath)
    # session2 = BehaviorOphysSession(789359614, api=api_2)
    
    # assert session == session2

    session = BehaviorOphysSession(789359614)
    # print(session.max_projection)
    # print(session.stimulus_timestamps)
    # print(session.ophys_timestamps)
    # print(session.metadata)
    # print(session.dff_traces)
    # print(session.cell_specimen_table)
    print(session.running_speed)
    # print(session.running_data_df)
    # print(session.stimulus_presentations)
    # print(session.stimulus_templates)
    # print(session.stimulus_index)
    # print(session.licks)
    # print(session.rewards)
    # print(session.task_parameters)
    # print(session.trials)
    # print(session.corrected_fluorescence_traces)
    # print(session.average_image)
    # print(session.motion_correction)
