import pytest

import copy
from allensdk.brain_observatory.behavior.\
    data_objects.eye_tracking.eye_tracking_table import EyeTrackingTable
from allensdk.brain_observatory.behavior.\
    data_objects.licks import Licks
from allensdk.brain_observatory.behavior.\
    data_objects.rewards import Rewards
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)


@pytest.mark.requires_bamboo
def test_incomplete_eye_tracking(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_eye_tracking = behavior_ophys_experiment_fixture.eye_tracking
    empty_eye_tracking = EyeTrackingTable.from_nwb(skeletal_nwb_fixture)
    empty_eye_tracking_df = empty_eye_tracking.value

    assert len(populated_eye_tracking) > 0
    assert len(empty_eye_tracking_df) == 0

    populated_columns = set(populated_eye_tracking.columns)
    empty_columns = set(empty_eye_tracking_df.columns)
    assert populated_columns == empty_columns

    assert (populated_eye_tracking.index.name
            == empty_eye_tracking_df.index.name)

    # make sure, when round-tripping the experiments, that the
    # populated experiment still writes out a populated data frame
    nwb1 = behavior_ophys_experiment_fixture.to_nwb()
    assert 'EyeTracking' in nwb1.acquisition.keys()
    roundtrip1 = BehaviorOphysExperiment.from_nwb(nwb1)
    assert len(roundtrip1.eye_tracking) > 0

    nwb2 = empty_eye_tracking.to_nwb(skeletal_nwb_fixture)
    assert 'EyeTracking' not in nwb2.acquisition.keys()
    roundtrip2 = EyeTrackingTable.from_nwb(nwb2)
    assert len(roundtrip2.value) == 0


@pytest.mark.requires_bamboo
def test_incomplete_eye_tracking_from_lims(
        behavior_ophys_experiment_fixture):
    """
    Compare a BehaviorOphysExperiment without eye tracking data
    from_lims with a BehaviorOphysExperiment with eye tracking data.
    Make sure the DataFrames have the same columns
    """

    incomplete_exp_id = 806456687
    incomplete_experiment = BehaviorOphysExperiment.from_lims(
        incomplete_exp_id,
        load_stimulus_movie=False)
    complete = behavior_ophys_experiment_fixture.eye_tracking
    incomplete = incomplete_experiment.eye_tracking

    assert len(complete) > 0
    assert len(incomplete) == 0

    assert set(complete.columns) == set(incomplete.columns)
    assert complete.index.name == incomplete.index.name


@pytest.mark.requires_bamboo
def test_incomplete_licks(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_licks = behavior_ophys_experiment_fixture.licks
    empty_licks = Licks.from_nwb(skeletal_nwb_fixture)
    empty_licks_df = empty_licks.value

    assert len(populated_licks) > 0
    assert len(empty_licks_df) == 0

    populated_columns = set(populated_licks.columns)
    empty_columns = set(empty_licks_df.columns)
    assert populated_columns == empty_columns

    assert populated_licks.index.name == empty_licks_df.index.name

    # make sure, when round-tripping the experiments, that the
    # populated experiment still writes out a populated data frame
    nwb1 = behavior_ophys_experiment_fixture.to_nwb()
    assert 'licking' in nwb1.processing.keys()
    roundtrip1 = BehaviorOphysExperiment.from_nwb(nwb1)
    assert len(roundtrip1.licks) > 0

    nwb2 = empty_licks.to_nwb(skeletal_nwb_fixture)
    assert 'licking' not in nwb2.processing.keys()
    roundtrip2 = Licks.from_nwb(nwb2)
    assert len(roundtrip2.value) == 0


@pytest.mark.requires_bamboo
def test_incomplete_rewards(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_rewards = behavior_ophys_experiment_fixture.rewards
    empty_rewards = Rewards.from_nwb(skeletal_nwb_fixture)
    empty_rewards_df = empty_rewards.value

    assert len(populated_rewards) > 0
    assert len(empty_rewards_df) == 0

    populated_columns = set(populated_rewards.columns)
    empty_columns = set(empty_rewards_df.columns)
    assert populated_columns == empty_columns

    assert populated_rewards.index.name == empty_rewards_df.index.name

    # make sure, when round-tripping the experiments, that the
    # populated experiment still writes out a populated data frame
    nwb1 = behavior_ophys_experiment_fixture.to_nwb()
    assert 'rewards' in nwb1.processing.keys()
    roundtrip1 = BehaviorOphysExperiment.from_nwb(nwb1)
    assert len(roundtrip1.rewards) > 0

    nwb2 = empty_rewards.to_nwb(skeletal_nwb_fixture)
    assert 'rewards' not in nwb2.processing.keys()
    roundtrip2 = Rewards.from_nwb(nwb2)
    assert len(roundtrip2.value) == 0


@pytest.mark.requires_bamboo
def test_incomplete_rig_geometry(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_rig_geom = behavior_ophys_experiment_fixture.\
                                eye_tracking_rig_geometry

    assert len(populated_rig_geom) > 0

    # copy the BehaviorOphysExperiment, set its rig geometry to
    # None by hand, write it out to an NWB file, read that NWB
    # file back, and make sure that an empty data object of the
    # correct type is returned for eye_tracking_rig_geometry

    no_rig_geom = copy.deepcopy(behavior_ophys_experiment_fixture)

    no_rig_geom._eye_tracking_rig_geometry = None

    # make sure we didn't alter the fixture
    assert behavior_ophys_experiment_fixture.\
           _eye_tracking_rig_geometry is not None

    nwb = no_rig_geom.to_nwb()
    test_ophys_experiment = BehaviorOphysExperiment.from_nwb(nwb)
    assert 'eye_tracking_rig_metadata' not in nwb.processing.keys()

    assert len(test_ophys_experiment.eye_tracking_rig_geometry) == 0

    assert isinstance(
            test_ophys_experiment.eye_tracking_rig_geometry,
            type(behavior_ophys_experiment_fixture.eye_tracking_rig_geometry))

    # make sure that the populated BehaviorOphysExperiment actually
    # writes out RigGeometry to NWB files
    nwb = behavior_ophys_experiment_fixture.to_nwb()
    assert 'eye_tracking_rig_metadata' in nwb.processing.keys()
    roundtrip = BehaviorOphysExperiment.from_nwb(nwb)
    assert len(roundtrip.eye_tracking_rig_geometry) > 0
