from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)

from allensdk.brain_observatory.session_api_utils import sessions_are_equal

import pytest
import pathlib
from pynwb import NWBHDF5IO

from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


@pytest.mark.requires_bamboo
def test_nwb_end_to_end_session(
        tmpdir_factory,
        helper_functions):
    session_id = 870987812
    tmpdir = tmpdir_factory.mktemp('session_nwb_end_to_end')
    tmpdir = pathlib.Path(tmpdir)
    nwb_path = tmpdir / f'session_{session_id}.nwb'
    session = BehaviorSession.from_lims(
            behavior_session_id=session_id,
            load_stimulus_movie=False)
    nwb_file = session.to_nwb()
    with NWBHDF5IO(nwb_path, 'w') as nwb_file_writer:
        nwb_file_writer.write(nwb_file)

    roundtrip = BehaviorSession.from_nwb_path(
            nwb_path=str(nwb_path.resolve().absolute()))

    assert sessions_are_equal(
            session,
            roundtrip,
            reraise=True)

    helper_functions.windows_safe_cleanup_dir(tmpdir)


@pytest.fixture
def session_data_fixture():

    return {
        "behavior_session_id": 1010991549,
        "foraging_id": "bfcc4803-8892-4cb4-88e0-9437b98936db",
        "driver_line": [
            "Vip-IRES-Cre"
        ],
        "reporter_line": [
            "Ai32(RCL-ChR2(H134R)_EYFP)"
        ],
        "full_genotype": "Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt",
        "rig_name": "BEH.G-Box1",
        "date_of_acquisition": "2020-02-28 03:11:17",
        "external_specimen_name": 506940,
        "behavior_stimulus_file":
            "/allen/programs/braintv/production/visualbehavior/prod0/"
            "specimen_1000324129/behavior_session_1010991549/"
            "200228111053_506940_bfcc4803-8892-4cb4-88e0-9437b98936db.pkl",
        "date_of_birth": "2019-11-24 16:00:00",
        "sex": "M",
        "age": "unknown",
        "stimulus_name": None
    }


def test_behavior_session_list_data_attributes_and_methods(monkeypatch):
    """Test that data related methods/attributes/properties for
    BehaviorSession are returned properly."""

    def dummy_init(self):
        pass

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorSession, '__init__', dummy_init)
        bs = BehaviorSession()
        obt = bs.list_data_attributes_and_methods()

    expected = {
        'behavior_session_id',
        'get_performance_metrics',
        'get_reward_rate',
        'get_rolling_performance_df',
        'licks',
        'metadata',
        'raw_running_speed',
        'rewards',
        'running_speed',
        'stimulus_presentations',
        'stimulus_templates',
        'stimulus_fingerprint_movie_template',
        'stimulus_timestamps',
        'task_parameters',
        'trials',
        'eye_tracking',
        'eye_tracking_rig_geometry'
    }

    assert any(expected ^ set(obt)) is False


@pytest.mark.nightly
def test_behavior_session_equivalent_json_lims(session_data_fixture):

    json_session = BehaviorSession.from_json(session_data_fixture)

    behavior_session_id = session_data_fixture['behavior_session_id']
    lims_session = BehaviorSession.from_lims(
        behavior_session_id,
        load_stimulus_movie=False
    )

    assert sessions_are_equal(json_session, lims_session, reraise=True)


class TestBehaviorSession(LimsTest):
    @pytest.mark.requires_bamboo
    def test_eye_tracking_loaded_with_metadata_frame(self):
        # This session uses MVR to record the eye tracking video
        sess_id = 1154034257

        sess = BehaviorSession.from_lims(behavior_session_id=sess_id,
                                         lims_db=self.dbconn,
                                         load_stimulus_movie=False)
        assert not sess.eye_tracking.empty
