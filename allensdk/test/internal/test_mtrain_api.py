import pytest

from allensdk.internal.api.mtrain_api import MtrainApi, MtrainSqlApi


@pytest.mark.nightly
@pytest.mark.parametrize(
    "api",
    [
        pytest.param(MtrainApi()),
        pytest.param(MtrainSqlApi()),
    ],
)
def test_get_subjects(api):
    subject_list = api.get_subjects()
    assert len(subject_list) > 190 and 423746 in subject_list


@pytest.mark.nightly
@pytest.mark.parametrize(
    "api",
    [
        pytest.param(MtrainApi()),
        pytest.param(MtrainSqlApi()),
    ],
)
def test_get_behavior_training_df(api):
    LabTracks_ID = 423986
    df = api.get_behavior_training_df(LabTracks_ID)
    # assert list(df.columns) == [u'stage_name', u'regimen_name', u'date',
    # u'behavior_session_id']
    assert len(df) == 24


@pytest.mark.nightly
@pytest.mark.parametrize(
    "LabTracks_ID",
    [
        pytest.param(423986),
    ],
)
def test_get_current_stage(LabTracks_ID):
    api = MtrainApi()
    stage = api.get_current_stage(LabTracks_ID)
    assert stage == "OPHYS_6_images_B"


@pytest.mark.nightly
@pytest.mark.parametrize(
    "behavior_session_uuid, behavior_session_id",
    [
        pytest.param("394a910e-94c7-4472-9838-5345aff59ed8", None),
        pytest.param(None, 823847007),
        pytest.param("394a910e-94c7-4472-9838-5345aff59ed8", 823847007),
    ],
)
def test_get_session(behavior_session_uuid, behavior_session_id):
    api = MtrainApi()
    kwargs = {
        key: val
        for key, val in [("behavior_session_uuid", behavior_session_uuid), ("behavior_session_id", behavior_session_id)]
        if val is not None
    }
    session_dict = api.get_session(**kwargs)
    trials_df = session_dict.pop("trials")
    assert len(trials_df) == 576
    assert "stages" in session_dict.keys()  # Remove stages because it's
    # very long
    del session_dict["stages"]
    assert session_dict == {
        "name": "TRAINING_1_gratings",
        "parameters": {
            "auto_reward_delay": 0.15,
            "change_time_scale": 2.0,
            "end_after_response": True,
            "change_flashes_max": None,
            "change_time_dist": "exponential",
            "stimulus_window": 6.0,
            "response_window": [0.15, 1.0],
            "change_flashes_min": None,
            "catch_frequency": 0.25,
            "min_no_lick_time": 0.0,
            "timeout_duration": 0.3,
            "free_reward_trials": 10,
            "volume_limit": 5.0,
            "max_task_duration_min": 60.0,
            "reward_volume": 0.01,
            "end_after_response_sec": 3.5,
            "start_stop_padding": 20.0,
            "periodic_flash": None,
            "stage": "TRAINING_1_gratings",
            "auto_reward_vol": 0.005,
            "task_id": "DoC",
            "stimulus": {
                "params": {"phase": 0.25, "tex": "sqr", "units": "deg", "sf": 0.04, "size": [200, 150]},
                "class": "grating",
                "groups": {"horizontal": {"Ori": [90, 270]}, "vertical": {"Ori": [0, 180]}},
            },
            "failure_repeats": 5,
            "warm_up_trials": 5,
            "pre_change_time": 2.25,
        },
        "script": "http://stash.corp.alleninstitute.org/"
        "projects/VB/repos/visual_behavior_scripts/"
        "raw/change_detection_with_fingerprint.py?at="
        "021ec55fbbdbb05aad1681c016e83066fe5aa1dd",
        "behavior_session_uuid": "394a910e-94c7-4472-9838-5345aff59ed8",
        "script_md5": "e0535f3b6f03ccc8eeccaed2118f3c1d",
        "LabTracks_ID": 431151,
        "date": "2019-02-15T13:01:23.672000",
        "regimen_name": "VisualBehavior_Task1A_v1.0.1",
        "default_x": False,
        "regimens": [{"active": False, "default": False, "id": 14, "name": "VisualBehavior_Task1A_v1.0.1"}],
        "default_y": False,
    }
