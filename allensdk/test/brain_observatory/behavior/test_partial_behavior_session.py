"""
This script contains tests to ensure that behavior_sessions that are
missing data objects return empty DataFrames with the same columns
as completely populated behavior_session
"""
import pytest
import pandas as pd
import numpy as np
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)
from allensdk.brain_observatory.behavior.image_api import Image


def dataframes_same_shape(df1: pd.DataFrame,
                          df2: pd.DataFrame):
    """
    Assert that df1 and df2 have the same column names
    and that their indexes have the same name
    """
    df1_c = df1.columns
    df2_c = df2.columns
    assert len(df1_c) == len(df2_c)
    df1_c_set = set(df1_c)
    for col_name in df2_c:
        assert col_name in df1_c_set
    if df1.index.name is None:
        assert df2.index.name is None
    else:
        assert df1.index.name == df2.index.name


def test_empty_behavior_session(
        empty_behavior_session_fixture):
    """
    Writing one massive test because instantiating the fixtures
    is apparently expensive
    """

    assert empty_behavior_session_fixture.licks.index.name is None
    expected = set(['timestamps', 'frame'])
    actual = set(empty_behavior_session_fixture.licks.columns)
    assert actual == expected

    assert empty_behavior_session_fixture.raw_running_speed.index.name is None
    expected = set(['timestamps', 'speed'])
    actual = set(empty_behavior_session_fixture.raw_running_speed.columns)
    assert actual == expected

    assert empty_behavior_session_fixture.rewards.index.name is None
    expected = set(['volume', 'timestamps', 'autorewarded'])
    actual = set(empty_behavior_session_fixture.rewards.columns)
    assert actual == expected

    assert empty_behavior_session_fixture.running_speed.index.name is None
    expected = set(['timestamps', 'speed'])
    actual = set(empty_behavior_session_fixture.running_speed.columns)
    assert actual == expected

    df = empty_behavior_session_fixture.stimulus_presentations
    assert df.index.name == 'stimulus_presentations_id'
    expected = set(['start_time', 'stop_time', 'duration',
                    'image_name', 'image_index', 'is_change',
                    'omitted', 'start_frame', 'end_frame', 'image_set'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_behavior_session_fixture.stimulus_templates
    assert df.index.name == 'image_name'
    expected = set(['unwarped', 'warped'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_behavior_session_fixture.trials
    assert df.index.name == 'trials_id'
    expected = set(['start_time', 'stop_time', 'lick_times',
                    'reward_time', 'reward_volume',
                    'hit', 'false_alarm', 'miss', 'stimulus_change',
                    'aborted', 'go', 'catch', 'auto_rewarded',
                    'correct_reject', 'trial_length',
                    'response_time', 'change_frame', 'change_time',
                    'response_latency', 'initial_image_name',
                    'change_image_name'])
    actual = set(df.columns)
    assert actual == expected

    # stimulus_timestamps
    assert isinstance(empty_behavior_session_fixture.stimulus_timestamps,
                      np.ndarray)

    # task_parameters
    assert isinstance(empty_behavior_session_fixture.task_parameters,
                      dict)

    # metadata
    assert isinstance(empty_behavior_session_fixture.metadata,
                      dict)


@pytest.mark.requires_bamboo
def test_compare_session_from_lims(empty_behavior_session_fixture):

    good_session_id = 948206919  # session with all data objects
    good_session = BehaviorSession.from_lims(
                                     behavior_session_id=good_session_id)

    # make sure that the empty session has dataframes with the same
    # column names as the complete session
    assert len(good_session.licks) > 0
    dataframes_same_shape(
        good_session.licks,
        empty_behavior_session_fixture.licks)

    assert len(good_session.raw_running_speed) > 0
    dataframes_same_shape(
        good_session.raw_running_speed,
        empty_behavior_session_fixture.raw_running_speed)

    assert len(good_session.rewards) > 0
    dataframes_same_shape(
        good_session.rewards,
        empty_behavior_session_fixture.rewards)

    assert len(good_session.running_speed) > 0
    dataframes_same_shape(
        good_session.running_speed,
        empty_behavior_session_fixture.running_speed)

    assert len(good_session.stimulus_presentations) > 0
    dataframes_same_shape(
        good_session.stimulus_presentations,
        empty_behavior_session_fixture.stimulus_presentations)

    assert len(good_session.stimulus_templates) > 0
    dataframes_same_shape(
        good_session.stimulus_templates,
        empty_behavior_session_fixture.stimulus_templates)

    assert len(good_session.trials) > 0
    dataframes_same_shape(
        good_session.trials,
        empty_behavior_session_fixture.trials)


def test_empty_ophys_experiment(
        empty_ophys_experiment_fixture):
    """
    Writing one massive test because instantiating the fixtures
    is apparently expensive
    """

    df = empty_ophys_experiment_fixture.cell_specimen_table
    assert df.index.name == 'cell_specimen_id'
    expected = set(['cell_roi_id', 'height', 'mask_image_plane',
                    'max_correction_down', 'max_correction_left',
                    'max_correction_right', 'max_correction_up',
                    'valid_roi', 'width', 'x', 'y', 'roi_mask'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.corrected_fluorescence_traces
    assert df.index.name == 'cell_specimen_id'
    expected = set (['cell_roi_id', 'corrected_fluorescence'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.dff_traces
    assert df.index.name == 'cell_specimen_id'
    expected = set(['cell_roi_id', 'dff'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.events
    assert df.index.name == 'cell_specimen_id'
    expected = set(['cell_roi_id', 'events', 'filtered_events',
                    'lambda', 'noise_std'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.eye_tracking
    assert df.index.name == 'frame'
    expected = set(['timestamps', 'cr_area', 'eye_area',
                    'pupil_area', 'likely_blink', 'pupil_area_raw',
                    'cr_area_raw', 'eye_area_raw', 'cr_center_x',
                    'cr_center_y', 'cr_width', 'cr_height', 'cr_phi',
                    'eye_center_x', 'eye_center_y', 'eye_width',
                    'eye_height', 'eye_phi', 'pupil_center_x',
                    'pupil_center_y', 'pupil_width', 'pupil_height',
                    'pupil_phi'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.motion_correction
    assert df.index.name is None
    expected = set(['x', 'y'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.roi_masks
    assert df.index.name == 'cell_specimen_id'
    expected = set(['cell_roi_id', 'roi_mask'])
    actual = set(df.columns)
    assert actual == expected

    assert empty_ophys_experiment_fixture.licks.index.name is None
    expected = set(['timestamps', 'frame'])
    actual = set(empty_ophys_experiment_fixture.licks.columns)
    assert actual == expected

    assert empty_ophys_experiment_fixture.raw_running_speed.index.name is None
    expected = set(['timestamps', 'speed'])
    actual = set(empty_ophys_experiment_fixture.raw_running_speed.columns)
    assert actual == expected

    assert empty_ophys_experiment_fixture.rewards.index.name is None
    expected = set(['volume', 'timestamps', 'autorewarded'])
    actual = set(empty_ophys_experiment_fixture.rewards.columns)
    assert actual == expected

    assert empty_ophys_experiment_fixture.running_speed.index.name is None
    expected = set(['timestamps', 'speed'])
    actual = set(empty_ophys_experiment_fixture.running_speed.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.stimulus_presentations
    assert df.index.name == 'stimulus_presentations_id'
    expected = set(['start_time', 'stop_time', 'duration',
                    'image_name', 'image_index', 'is_change',
                    'omitted', 'start_frame', 'end_frame', 'image_set'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.stimulus_templates
    assert df.index.name == 'image_name'
    expected = set(['unwarped', 'warped'])
    actual = set(df.columns)
    assert actual == expected

    df = empty_ophys_experiment_fixture.trials
    assert df.index.name == 'trials_id'
    expected = set(['start_time', 'stop_time', 'lick_times',
                    'reward_time', 'reward_volume',
                    'hit', 'false_alarm', 'miss', 'stimulus_change',
                    'aborted', 'go', 'catch', 'auto_rewarded',
                    'correct_reject', 'trial_length',
                    'response_time', 'change_frame', 'change_time',
                    'response_latency', 'initial_image_name',
                    'change_image_name'])
    actual = set(df.columns)
    assert actual == expected

    # stimulus_timestamps
    assert isinstance(empty_ophys_experiment_fixture.stimulus_timestamps,
                      np.ndarray)

    # task_parameters
    assert isinstance(empty_ophys_experiment_fixture.task_parameters,
                      dict)

    empty_image = Image(data=np.ones(0), spacing=tuple())
    assert empty_ophys_experiment_fixture.max_projection == empty_image
    assert empty_ophys_experiment_fixture.average_projection == empty_image
    img = empty_ophys_experiment_fixture.segmentation_mask_image
    assert img == empty_image

    timestamps = empty_ophys_experiment_fixture.ophys_timestamps
    np.testing.assert_array_equal(timestamps, np.ones(0))


@pytest.mark.requires_bamboo
def test_compare_ophys_exp_from_lims(
        empty_ophys_experiment_fixture):

    baseline_exp = BehaviorOphysExperiment.from_lims(
                          ophys_experiment_id = 953443028)
    test_exp = empty_ophys_experiment_fixture

    assert not baseline_exp.max_projection == test_exp.max_projection
    assert isinstance(test_exp.max_projection,
                      type(baseline_exp.max_projection))

    assert not baseline_exp.average_projection == test_exp.average_projection
    assert isinstance(test_exp.average_projection,
                      type(baseline_exp.average_projection))

    img = baseline_exp.segmentation_mask_image
    assert not img == test_exp.segmentation_mask_image
    assert isinstance(test_exp.segmentation_mask_image,
                      type(img))

    assert not np.array_equal(baseline_exp.ophys_timestamps,
                              test_exp.ophys_timestamps)
    assert isinstance(test_exp.ophys_timestamps,
                       type(baseline_exp.ophys_timestamps))

    for df_pair in [(baseline_exp.cell_specimen_table,
                     test_exp.cell_specimen_table),
                    (baseline_exp.corrected_fluorescence_traces,
                     test_exp.corrected_fluorescence_traces),
                    (baseline_exp.dff_traces, test_exp.dff_traces),
                    (baseline_exp.events, test_exp.events),
                    (baseline_exp.eye_tracking, test_exp.eye_tracking),
                    (baseline_exp.licks, test_exp.licks),
                    (baseline_exp.motion_correction,
                     test_exp.motion_correction),
                    (baseline_exp.raw_running_speed,
                     test_exp.raw_running_speed),
                    (baseline_exp.rewards, test_exp.rewards),
                    (baseline_exp.roi_masks, test_exp.roi_masks),
                    (baseline_exp.running_speed, test_exp.running_speed),
                    (baseline_exp.stimulus_presentations,
                     test_exp.stimulus_presentations),
                    (baseline_exp.stimulus_templates,
                     test_exp.stimulus_templates),
                    (baseline_exp.trials, test_exp.trials)]:

        baseline_df = df_pair[0]
        test_df = df_pair[1]
        assert len(baseline_df) > 0
        assert len(test_df) == 0
        dataframes_same_shape(test_df, baseline_df)
