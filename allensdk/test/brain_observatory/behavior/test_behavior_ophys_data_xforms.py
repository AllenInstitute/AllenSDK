import pytest
import logging
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.session_apis.data_transforms import BehaviorOphysDataTransforms  # noqa: E501
from allensdk.internal.brain_observatory.time_sync import OphysTimeAligner
from allensdk.test.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.test_behavior_metadata import \
    TestBehaviorMetadata


@pytest.mark.parametrize("roi_ids,expected", [
    [
        1,
        np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    ],
    [
        None,
        np.array([
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
    ]
])
# cell_specimen_table_api fixture from allensdk.test.brain_observatory.conftest
def test_get_roi_masks_by_cell_roi_id(roi_ids, expected,
                                      cell_specimen_table_api):
    api = cell_specimen_table_api
    obtained = api.get_roi_masks_by_cell_roi_id(roi_ids)
    assert np.allclose(expected, obtained.values)
    assert np.allclose(obtained.coords['row'],
                       [0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(obtained.coords['column'],
                       [0.25, 0.75, 1.25, 1.75, 2.25])


def test_get_rewards(monkeypatch):
    """
    Test that BehaviorOphysDataTransforms.get_rewards() returns
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
        ctx.setattr(BehaviorOphysDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorOphysDataTransforms,
                    'get_stimulus_timestamps',
                    dummy_stimulus_timestamps)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xforms = BehaviorOphysDataTransforms()

        rewards = xforms.get_rewards()

        expected_dict = {'volume': [0.001, 0.002],
                         'timestamps': [0.04, 0.1],
                         'autorewarded': [True, False]}
        expected_df = pd.DataFrame(expected_dict)
        expected_df = expected_df
        assert expected_df.equals(rewards)


def test_get_licks(monkeypatch):
    """
    Test that BehaviorOphysDataTransforms.get_licks() a dataframe
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
        ctx.setattr(BehaviorOphysDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorOphysDataTransforms,
                    'get_stimulus_timestamps',
                    dummy_stimulus_timestamps)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xforms = BehaviorOphysDataTransforms()

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


def test_get_licks_excess(monkeypatch):
    """
    Test that BehaviorOphysDataTransforms.get_licks() in the case where
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
        ctx.setattr(BehaviorOphysDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorOphysDataTransforms,
                    'get_stimulus_timestamps',
                    dummy_stimulus_timestamps)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xforms = BehaviorOphysDataTransforms()

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
    Test that BehaviorOphysDataTransforms.get_licks() in the case where
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
        ctx.setattr(BehaviorOphysDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorOphysDataTransforms,
                    'get_stimulus_timestamps',
                    dummy_stimulus_timestamps)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xforms = BehaviorOphysDataTransforms()

        licks = xforms.get_licks()

        expected_dict = {'timestamps': [],
                         'frame': []}
        expected_df = pd.DataFrame(expected_dict)
        assert expected_df.columns.equals(licks.columns)
        np.testing.assert_array_equal(expected_df.timestamps.to_numpy(),
                                      licks.timestamps.to_numpy())
        np.testing.assert_array_equal(expected_df.frame.to_numpy(),
                                      licks.frame.to_numpy())


def test_get_licks_failure(monkeypatch):
    """
    Test that BehaviorOphysDataTransforms.get_licks() fails if the last lick
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
        ctx.setattr(BehaviorOphysDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorOphysDataTransforms,
                    'get_stimulus_timestamps',
                    dummy_stimulus_timestamps)

        ctx.setattr(BehaviorOphysDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xforms = BehaviorOphysDataTransforms()
        with pytest.raises(IndexError):
            xforms.get_licks()


def test_timestamps_and_delay(monkeypatch):
    """
    Test that BehaviorOphysDataTransforms returns the right values
    with get_stimulus_timestamps and get_monitor_delay
    """
    def dummy_loader(self):
        self._stimulus_timestamps = np.array([2, 3, 7])
        self._monitor_delay = 99.3

    def dummy_init(self):
        pass

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysDataTransforms,
                    "__init__",
                    dummy_init)
        ctx.setattr(BehaviorOphysDataTransforms,
                    "_load_stimulus_timestamps_and_delay",
                    dummy_loader)

        xforms = BehaviorOphysDataTransforms()
        np.testing.assert_array_equal(xforms.get_stimulus_timestamps(),
                                      np.array([2, 3, 7]))
        assert abs(xforms.get_monitor_delay() - 99.3) < 1.0e-10

        # need to reverse order to make sure loader works
        # correctly
        xforms = BehaviorOphysDataTransforms()
        assert abs(xforms.get_monitor_delay() - 99.3) < 1.0e-10
        np.testing.assert_array_equal(xforms.get_stimulus_timestamps(),
                                      np.array([2, 3, 7]))
