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
