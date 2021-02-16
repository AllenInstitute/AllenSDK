import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.session_apis.data_transforms import BehaviorDataTransforms  # noqa: E501


def test_get_stimulus_timestamps(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_stimulus_timestamps()
    just returns the sum of the intervalsms field in the
    behavior stimulus pickle file, padded with a zero at the
    first timestamp.
    """

    expected = np.array([0., 0.0001, 0.0003, 0.0006, 0.001])

    def dummy_init(self):
        pass

    def dummy_stimulus_file(self):
        intervalsms = [0.1, 0.2, 0.3, 0.4]
        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['intervalsms'] = intervalsms
        return data

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xform = BehaviorDataTransforms()
        timestamps = xform.get_stimulus_timestamps()
        np.testing.assert_array_almost_equal(timestamps,
                                             expected,
                                             decimal=10)


def test_get_rewards(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_rewards() returns
    expected results (main nuance is that timestamps should be
    determined by applying the reward frame as an index to
    stimulus_timestamps)
    """

    def dummy_init(self):
        pass

    def dummy_stimulus_timestamps(self):
        return np.arange(0, 2.0, 0.01)

    def dummy_stimulus_file(self):
        trial_log = []
        trial_log.append({'rewards': [(0.001, -1.0, 4)],
                          'trial_params': {'auto_reward': True}})
        trial_log.append({'rewards': []})
        trial_log.append({'rewards': [(0.002, -1.0, 10)],
                          'trial_params': {'auto_reward': False}})
        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['trial_log'] = trial_log
        return data

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorDataTransforms,
                    'get_stimulus_timestamps',
                    dummy_stimulus_timestamps)

        ctx.setattr(BehaviorDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xforms = BehaviorDataTransforms()

        rewards = xforms.get_rewards()

        expected_dict = {'volume': [0.001, 0.002],
                         'timestamps': [0.04, 0.1],
                         'autorewarded': [True, False]}
        expected_df = pd.DataFrame(expected_dict)
        expected_df = expected_df.set_index('timestamps', drop=True)
        assert expected_df.equals(rewards)
