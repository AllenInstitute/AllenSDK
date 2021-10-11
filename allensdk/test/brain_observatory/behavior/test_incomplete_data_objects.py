import pytest

from allensdk.brain_observatory.behavior.\
    data_objects.eye_tracking.eye_tracking_table import EyeTrackingTable
from allensdk.brain_observatory.behavior.\
    data_objects.licks import Licks
from allensdk.brain_observatory.behavior.\
    data_objects.rewards import Rewards


@pytest.mark.requires_bamboo
def test_incomplete_eye_tracking(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_eye_tracking = behavior_ophys_experiment_fixture.eye_tracking
    empty_eye_tracking = EyeTrackingTable.from_nwb(skeletal_nwb_fixture).value

    assert len(populated_eye_tracking) > 0
    assert len(empty_eye_tracking) == 0

    populated_columns = set(populated_eye_tracking.columns)
    empty_columns = set(empty_eye_tracking.columns)
    assert populated_columns == empty_columns

    assert populated_eye_tracking.index.name == empty_eye_tracking.index.name


@pytest.mark.requires_bamboo
def test_incomplete_licks(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_licks = behavior_ophys_experiment_fixture.licks
    empty_licks = Licks.from_nwb(skeletal_nwb_fixture).value

    assert len(populated_licks) > 0
    assert len(empty_licks) == 0

    populated_columns = set(populated_licks.columns)
    empty_columns = set(empty_licks.columns)
    assert populated_columns == empty_columns

    assert populated_licks.index.name == empty_licks.index.name


@pytest.mark.requires_bamboo
def test_incomplete_rewards(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_rewards = behavior_ophys_experiment_fixture.rewards
    empty_rewards = Rewards.from_nwb(skeletal_nwb_fixture).value

    assert len(populated_rewards) > 0
    assert len(empty_rewards) == 0

    populated_columns = set(populated_rewards.columns)
    empty_columns = set(empty_rewards.columns)
    assert populated_columns == empty_columns

    assert populated_rewards.index.name == empty_rewards.index.name
