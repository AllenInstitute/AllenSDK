from pathlib import Path

import pynwb
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

    # Don't load LFP here to speed up the tests
    for probe in config['probes']:
        probe['lfp'] = None
    return BehaviorEcephysSession.from_json(
        session_data=config,
        skip_probes=['probeB', 'probeC', 'probeD', 'probeE', 'probeF']
    )


@pytest.fixture(scope='module')
def behavior_ecephys_session_with_lfp_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return a BehaviorEcephysSession for testing
    """
    config = copy.deepcopy(behavior_ecephys_session_config_fixture)

    return BehaviorEcephysSession.from_json(
        session_data=config,
        skip_probes=['probeB', 'probeC', 'probeD', 'probeE', 'probeF']
    )


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip', [True, False])
def test_read_write_session_nwb(
        roundtrip,
        data_object_roundtrip_fixture,
        behavior_ecephys_session_fixture):
    """Tests roundtrip of the session data"""
    nwbfile, _ = behavior_ecephys_session_fixture.to_nwb()

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=BehaviorEcephysSession)
    else:
        obt = BehaviorEcephysSession.from_nwb(nwbfile=nwbfile)

    assert obt == behavior_ecephys_session_fixture


@pytest.mark.requires_bamboo
def test_read_write_session_with_probe_nwb(
        data_object_roundtrip_fixture,
        behavior_ecephys_session_with_lfp_fixture,
        tmpdir
):
    """Tests roundtrip of a session with separate probe nwb files that store
    LFP and CSD data"""
    nwbfile, probe_nwbfile_map = \
        behavior_ecephys_session_with_lfp_fixture.to_nwb()

    probe_data_path_map = dict()
    for probe_name, probe_nwbfile in probe_nwbfile_map.items():
        path = Path(tmpdir) / f'probe_{probe_name}_lfp.nwb'
        with pynwb.NWBHDF5IO(path, 'w') as write_io:
            write_io.write(probe_nwbfile)
        probe_data_path_map[probe_name] = path

    obt = data_object_roundtrip_fixture(
        nwbfile=nwbfile,
        data_object_cls=BehaviorEcephysSession,
        probe_data_path_map=probe_data_path_map
    )

    # Load the LFP data into memory
    for probe in obt._probes:
        obt.get_lfp(probe_id=probe.id)
        assert probe.lfp is not None

    assert obt == behavior_ecephys_session_with_lfp_fixture


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
    trials_frames = trials[trials.is_change].change_frame
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


@pytest.mark.requires_bamboo
def test_getters_sanity_from_nwb(
        behavior_ecephys_session_fixture):
    """Sanity check to make sure that the BehaviorEcephysSession
    can use the BehaviorSession base class getter methods when read from nwb
    """
    nwbfile, _ = behavior_ecephys_session_fixture.to_nwb()

    sess = BehaviorEcephysSession.from_nwb(nwbfile=nwbfile)
    sess.get_performance_metrics()
    sess.get_rolling_performance_df()
    sess.get_reward_rate()
