import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

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
        self.roi_metrics = LazyProperty(self.api.get_roi_metrics, ophys_experiment_id=self.ophys_experiment_id)
        self.cell_roi_ids = LazyProperty(self.api.get_cell_roi_ids, ophys_experiment_id=self.ophys_experiment_id)
        self.running_speed = LazyProperty(self.api.get_running_speed, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.stimulus_table = LazyProperty(self.api.get_stimulus_table, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.stimulus_template = LazyProperty(self.api.get_stimulus_template, ophys_experiment_id=self.ophys_experiment_id)
        self.stimulus_metadata = LazyProperty(self.api.get_stimulus_metadata, ophys_experiment_id=self.ophys_experiment_id)
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
                elif isinstance(x1, (dict, list)):
                    assert x1 == x2
                else:
                    raise Exception('Comparator not implemented')

        except NotImplementedError as e:
            self_implements_get_field = hasattr(self.api, getattr(type(self), field).getter_name)
            other_implements_get_field = hasattr(other.api, getattr(type(other), field).getter_name)
            assert self_implements_get_field == other_implements_get_field == False

        except (AssertionError, AttributeError) as e:
            return False

        return True


if __name__ == "__main__":

    session = BehaviorOphysSession(789359614)

    from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
    nwb_filepath = './tmp.nwb'
    nwb_api = BehaviorOphysNwbApi(nwb_filepath)
    nwb_api.save(session)




    # print session.max_projection
    # print session.stimulus_timestamps
    # print session.ophys_timestamps
    # print session.metadata
    # print session.dff_traces
    # print session.roi_metrics
    # print session.cell_roi_ids
    # print session.running_speed
    # print session.stimulus_table
    # print session.stimulus_template
    # print session.stimulus_metadata
    # print session.licks
    # print session.rewards
    # print session.task_parameters
    # print session.trials
    # print session.corrected_fluorescence_traces
    # print session.average_image
    # print session.motion_correction