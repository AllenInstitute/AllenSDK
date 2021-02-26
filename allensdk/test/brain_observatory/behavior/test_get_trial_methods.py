"""
The tests in this file are little better than smoke tests.
They monkeypatch BehaviorDataTransforms and BehaviorOphysDataTransforms
to read the behavior_stimulus_pickle_file from

/resources/example_stimulus.pkl.gz

which is a copy of the pickle file for behavior session 891635659,
normally found at

/allen/programs/braintv/production/visualbehavior/prod2/specimen_850862430/behavior_session_891635659/190620111806_457841_c428be61-87d2-44f6-b00d-3401f28fa201.pkl

and just attempt to run get_trials() to make sure that passes safely
through that method. A more thorough testing of this method will require
a significant amount of work, as there are many edge cases and the
documentation of the stimulus pickle file is sparse. That should be the
focus of a future ticket.
"""

import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.session_apis.data_transforms import BehaviorDataTransforms  # noqa: E501
from allensdk.brain_observatory.behavior.session_apis.data_transforms import BehaviorOphysDataTransforms  # noqa: E501


def test_behavior_get_trials(monkeypatch):

    this_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(this_dir, 'resources')
    pkl_name = os.path.join(resource_dir, 'example_stimulus.pkl.gz')
    pkl_data = pd.read_pickle(pkl_name)

    def dummy_init(self):
        pass

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorDataTransforms,
                    '__init__',
                    dummy_init)
        ctx.setattr(BehaviorDataTransforms,
                    '_behavior_stimulus_file',
                    lambda x: pkl_data)

        xforms = BehaviorDataTransforms()
        _ = xforms.get_trials()


def test_behavior_ophys_get_trials(monkeypatch):

    this_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(this_dir, 'resources')
    pkl_name = os.path.join(resource_dir, 'example_stimulus.pkl.gz')
    pkl_data = pd.read_pickle(pkl_name)

    def dummy_init(self):
        pass

    n_t = len(pkl_data['items']['behavior']['intervalsms']) + 1
    timestamps = np.linspace(0, 1, n_t)

    def dummy_loader(self):
        self._stimulus_timestamps = np.copy(timestamps)
        self._monitor_delay = 0.021

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_load_stimulus_timestamps_and_delay',
                    dummy_loader)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_behavior_stimulus_file',
                    lambda x: pkl_data)

        xforms = BehaviorOphysDataTransforms()
        _ = xforms.get_trials()
