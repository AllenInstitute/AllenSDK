import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from typing import NamedTuple

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi


class BehaviorOphysSession(LazyPropertyMixin):

    def __init__(self, ophys_experiment_id, api=None, use_acq_trigger=False):

        self.ophys_experiment_id = ophys_experiment_id
        self.api = BehaviorOphysLimsApi() if api is None else api
        self.use_acq_trigger = use_acq_trigger

        self.max_projection = LazyProperty(self.api.get_max_projection, ophys_experiment_id=self.ophys_experiment_id)
        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.metadata = LazyProperty(self.api.get_metadata, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.dff_traces = LazyProperty(self.api.get_dff_traces, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.cell_specimen_table = LazyProperty(self.api.get_cell_specimen_table, ophys_experiment_id=self.ophys_experiment_id)
        self.running_speed = LazyProperty(self.api.get_running_speed, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.running_data_df = LazyProperty(self.api.get_running_data_df, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.stimulus_presentations = LazyProperty(self.api.get_stimulus_presentations, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.stimulus_templates = LazyProperty(self.api.get_stimulus_templates, ophys_experiment_id=self.ophys_experiment_id)
        self.stimulus_index = LazyProperty(self.api.get_stimulus_index, ophys_experiment_id=self.ophys_experiment_id)
        self.licks = LazyProperty(self.api.get_licks, ophys_experiment_id=self.ophys_experiment_id)
        self.rewards = LazyProperty(self.api.get_rewards, ophys_experiment_id=self.ophys_experiment_id)
        self.task_parameters = LazyProperty(self.api.get_task_parameters, ophys_experiment_id=self.ophys_experiment_id)
        self.trials = LazyProperty(self.api.get_trials, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.average_image = LazyProperty(self.api.get_average_image, ophys_experiment_id=self.ophys_experiment_id)
        self.motion_correction = LazyProperty(self.api.get_motion_correction, ophys_experiment_id=self.ophys_experiment_id)

    def __eq__(self, other):

        field_set = set()
        for key, val in self.__dict__.items():
            if isinstance(val, LazyProperty):
                field_set.add(key)
        for key, val in other.__dict__.items():
            if isinstance(val, LazyProperty):
                field_set.add(key)

        try:
            for field in field_set:
                x1, x2 = getattr(self, field), getattr(other, field)
                if isinstance(x1, pd.DataFrame):
                    assert_frame_equal(x1, x2)
                elif isinstance(x1, np.ndarray):
                    np.testing.assert_array_almost_equal(x1, x2)
                elif isinstance(x1, (list,)):
                    assert x1 == x2
                elif isinstance(x1, (dict,)):
                    for key in set(x1.keys()).union(set(x2.keys())):
                        if isinstance(x1[key], (np.ndarray,)):
                            np.testing.assert_array_almost_equal(x1[key], x2[key])
                        else:
                            assert x1[key] == x2[key]
                else:
                    assert x1 == x2

        except NotImplementedError as e:
            self_implements_get_field = hasattr(self.api, getattr(type(self), field).getter_name)
            other_implements_get_field = hasattr(other.api, getattr(type(other), field).getter_name)
            assert self_implements_get_field == other_implements_get_field == False

        except (AssertionError, AttributeError) as e:
            return False

        return True


if __name__ == "__main__":

    # from allensdk.brain_observatory import JSONEncoder
    # import json

    from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
    nwb_filepath = './tmp.nwb'
    nwb_api = BehaviorOphysNwbApi(nwb_filepath)
    session = BehaviorOphysSession(789359614)
    nwb_api.save(session)

    # session = BehaviorOphysSession(789359614)
    # print(session.max_projection)
    # print(session.stimulus_timestamps)
    # print(session.ophys_timestamps)
    # print(json.dumps(session.metadata, indent=2, cls=JSONEncoder))
    # print(session.dff_traces.head())
    # print(session.cell_specimen_table['image_mask'])
    # print(session.running_speed)
    # print(session.running_data_df)
    # print(session.stimulus_presentations)
    # print(session.stimulus_templates)
    # print(session.stimulus_index)
    # print(session.licks)
    # print(session.rewards)
    # print(json.dumps(session.task_parameters, indent=2, cls=JSONEncoder))
    # print(session.trials)
    # for key, val in list(session.trials.iterrows())[0][1].to_dict().items():
    #     print(key, val)
    # print(session.corrected_fluorescence_traces)
    # print(session.average_image)
    # print(session.motion_correction)



    # def get_task_parameters(self, *args, **kwargs):
    #     return read_data_json(self.task_parameters_file_info, object_hook=date_hook)

    # def save_task_parameters(self, obj):
    #     save_data_json(obj.task_parameters, self.task_parameters_file_info, cls=DateTimeEncoder)