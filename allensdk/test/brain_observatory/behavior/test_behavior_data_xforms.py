import pytest
import logging
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
        expected_df = expected_df
        assert expected_df.equals(rewards)


def test_get_licks(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_licks() a dataframe
    of licks whose timestamps are based on their frame number
    with respect to the stimulus_timestamps
    """

    def dummy_init(self):
        pass

    def dummy_stimulus_timestamps(self):
        return np.arange(0, 2.0, 0.01)

    def dummy_stimulus_file(self):

        # in this test, the trial log exists to make sure
        # that get_licks is *not* reading the licks from
        # here
        trial_log = []
        trial_log.append({'licks': [(-1.0, 100), (-1.0, 200)]})
        trial_log.append({'licks': [(-1.0, 300), (-1.0, 400)]})
        trial_log.append({'licks': [(-1.0, 500), (-1.0, 600)]})

        lick_events = [12, 15, 90, 136]
        lick_events = [{'lick_events': lick_events}]

        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['trial_log'] = trial_log
        data['items']['behavior']['lick_sensors'] = lick_events
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

        licks = xforms.get_licks()

        expected_dict = {'timestamps': [0.12, 0.15, 0.90, 1.36],
                         'frame': [12, 15, 90, 136]}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.columns)
        np.testing.assert_array_almost_equal(expected_df.timestamps.to_numpy(),
                                             licks.timestamps.to_numpy(),
                                             decimal=10)
        np.testing.assert_array_almost_equal(expected_df.frame.to_numpy(),
                                             licks.frame.to_numpy(),
                                             decimal=10)


def test_empty_licks(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_licks() in the case where
    there are no licks
    """

    def dummy_init(self):
        self.logger = logging.getLogger('dummy')
        pass

    def dummy_stimulus_timestamps(self):
        return np.arange(0, 2.0, 0.01)

    def dummy_stimulus_file(self):

        # in this test, the trial log exists to make sure
        # that get_licks is *not* reading the licks from
        # here
        trial_log = []
        trial_log.append({'licks': [(-1.0, 100), (-1.0, 200)]})
        trial_log.append({'licks': [(-1.0, 300), (-1.0, 400)]})
        trial_log.append({'licks': [(-1.0, 500), (-1.0, 600)]})

        lick_events = []
        lick_events = [{'lick_events': lick_events}]

        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['trial_log'] = trial_log
        data['items']['behavior']['lick_sensors'] = lick_events
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

        licks = xforms.get_licks()

        expected_dict = {'timestamps': [],
                         'frame': []}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.columns)
        np.testing.assert_array_equal(expected_df.timestamps.to_numpy(),
                                      licks.timestamps.to_numpy())
        np.testing.assert_array_equal(expected_df.frame.to_numpy(),
                                      licks.frame.to_numpy())


def test_get_licks_excess(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_licks() in the case where
    there is an extra frame at the end of the trial log and the mouse
    licked on that frame

    https://github.com/AllenInstitute/visual_behavior_analysis/blob/master/visual_behavior/translator/foraging2/extract.py#L640-L647
    """

    def dummy_init(self):
        self.logger = logging.getLogger('dummy')
        pass

    def dummy_stimulus_timestamps(self):
        return np.arange(0, 2.0, 0.01)

    def dummy_stimulus_file(self):

        # in this test, the trial log exists to make sure
        # that get_licks is *not* reading the licks from
        # here
        trial_log = []
        trial_log.append({'licks': [(-1.0, 100), (-1.0, 200)]})
        trial_log.append({'licks': [(-1.0, 300), (-1.0, 400)]})
        trial_log.append({'licks': [(-1.0, 500), (-1.0, 600)]})

        lick_events = [12, 15, 90, 136, 200]  # len(timestamps) == 200
        lick_events = [{'lick_events': lick_events}]

        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['trial_log'] = trial_log
        data['items']['behavior']['lick_sensors'] = lick_events
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

        licks = xforms.get_licks()

        expected_dict = {'timestamps': [0.12, 0.15, 0.90, 1.36],
                         'frame': [12, 15, 90, 136]}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.columns)
        np.testing.assert_array_almost_equal(expected_df.timestamps.to_numpy(),
                                             licks.timestamps.to_numpy(),
                                             decimal=10)
        np.testing.assert_array_almost_equal(expected_df.frame.to_numpy(),
                                             licks.frame.to_numpy(),
                                             decimal=10)


def test_get_licks_failure(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_licks() fails if the last lick
    is more than one frame beyond the end of the timestamps
    """

    def dummy_init(self):
        self.logger = logging.getLogger('dummy')
        pass

    def dummy_stimulus_timestamps(self):
        return np.arange(0, 2.0, 0.01)

    def dummy_stimulus_file(self):

        # in this test, the trial log exists to make sure
        # that get_licks is *not* reading the licks from
        # here
        trial_log = []
        trial_log.append({'licks': [(-1.0, 100), (-1.0, 200)]})
        trial_log.append({'licks': [(-1.0, 300), (-1.0, 400)]})
        trial_log.append({'licks': [(-1.0, 500), (-1.0, 600)]})

        lick_events = [12, 15, 90, 136, 201]  # len(timestamps) == 200
        lick_events = [{'lick_events': lick_events}]

        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['trial_log'] = trial_log
        data['items']['behavior']['lick_sensors'] = lick_events
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
        with pytest.raises(IndexError):
            xforms.get_licks()
