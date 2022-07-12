import pytest
import numpy as np
import copy

from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession


@pytest.fixture(scope='module')
def behavior_ecephys_session_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return a BehaviorEcephysSession for testing
    """
    config = copy.deepcopy(behavior_ecephys_session_config_fixture)
    config['probes'] = config['probes'][:3]
    return BehaviorEcephysSession.from_json(
        session_data=config)


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip', [True, False])
def test_read_write_nwb(roundtrip,
                        data_object_roundtrip_fixture,
                        behavior_ecephys_session_fixture):
    nwbfile = behavior_ecephys_session_fixture.to_nwb()

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=BehaviorEcephysSession)
    else:
        obt = BehaviorEcephysSession.from_nwb(nwbfile=nwbfile)

    assert obt == behavior_ecephys_session_fixture


@pytest.mark.requires_bamboo
def test_session_consistency(
        behavior_ecephys_session_fixture):
    """
    This method will test the self-consistency of
    the BehaviorEcephysSession
    """

    # test that the trials and stimulus_presentations tables
    # agree on the change_frames
    stim = behavior_ecephys_session_fixture.stimulus_presentations
    trials = behavior_ecephys_session_fixture.trials
    stim_frames = stim[stim.is_change & stim.active].start_frame
    trials_frames = trials[trials.stimulus_change].change_frame
    delta = stim_frames.values-trials_frames.values
    np.testing.assert_array_equal(
        delta,
        np.zeros(len(delta), dtype=int))

    # make sure that response_latency is not in the trials table
    assert 'response_latency' not in trials.columns


@pytest.mark.requires_bamboo
def test_getters_sanity(behavior_ecephys_session_fixture):
    """Sanity check to make sure that the BehaviorEcephysSession
    can use the BehaviorSession base class getter methods
    """
    behavior_ecephys_session_fixture.get_performance_metrics()
    behavior_ecephys_session_fixture.get_rolling_performance_df()
    behavior_ecephys_session_fixture.get_reward_rate()
