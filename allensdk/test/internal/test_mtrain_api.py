import pytest

from allensdk.internal.api import OneResultExpectedError
from allensdk.internal.api.mtrain_api import MtrainApi, MtrainSqlApi


@pytest.mark.nightly
@pytest.mark.parametrize('api', [
    pytest.param(MtrainApi()),
    pytest.param(MtrainSqlApi()),
])
def test_get_subjects(api):
    subject_list = api.get_subjects()
    assert len(subject_list) > 190 and 423746 in subject_list


@pytest.mark.nightly
@pytest.mark.parametrize('api', [
    pytest.param(MtrainApi()),
    pytest.param(MtrainSqlApi()),
])
def test_get_behavior_training_df(api):
    LabTracks_ID = 423986
    df = api.get_behavior_training_df(LabTracks_ID)
    # assert list(df.columns) == [u'stage_name', u'regimen_name', u'date', u'behavior_session_id']
    assert len(df) == 24


@pytest.mark.nightly
@pytest.mark.parametrize('LabTracks_ID', [
    pytest.param(423986),
])
def test_get_current_stage(LabTracks_ID):
    api = MtrainApi()
    stage = api.get_current_stage(LabTracks_ID)
    assert stage == 'OPHYS_6_images_B'


@pytest.mark.nightly
@pytest.mark.parametrize('behavior_session_uuid, behavior_session_id', 
    [pytest.param('394a910e-94c7-4472-9838-5345aff59ed8', None),
     pytest.param(None, 823847007),
     pytest.param('394a910e-94c7-4472-9838-5345aff59ed8', 823847007),
])
def test_get_session(behavior_session_uuid, behavior_session_id):

    api = MtrainApi()
    kwargs = {key:val for key, val in [('behavior_session_uuid', behavior_session_uuid), ('behavior_session_id', behavior_session_id)] if val is not None}
    session_dict = api.get_session(**kwargs)
    trials_df = session_dict.pop('trials')
    assert len(trials_df) == 576
    assert "stages" in session_dict.keys()    # Remove stages because it's very long
    del session_dict["stages"]
    assert session_dict == {u'name': u'TRAINING_1_gratings',
                                                      u'parameters': {u'auto_reward_delay': 0.15,
                                                                      u'change_time_scale': 2.0,
                                                                      u'end_after_response': True,
                                                                      u'change_flashes_max': None,
                                                                      u'change_time_dist': u'exponential',
                                                                      u'stimulus_window': 6.0,
                                                                      u'response_window': [0.15, 1.0],
                                                                      u'change_flashes_min': None,
                                                                      u'catch_frequency': 0.25,
                                                                      u'min_no_lick_time': 0.0,
                                                                      u'timeout_duration': 0.3,
                                                                      u'free_reward_trials': 10,
                                                                      u'volume_limit': 5.0,
                                                                      u'max_task_duration_min': 60.0,
                                                                      u'reward_volume': 0.01,
                                                                      u'end_after_response_sec': 3.5,
                                                                      u'start_stop_padding': 20.0,
                                                                      u'periodic_flash': None,
                                                                      u'stage': u'TRAINING_1_gratings',
                                                                      u'auto_reward_vol': 0.005,
                                                                      u'task_id': u'DoC',
                                                                      u'stimulus': {u'params': {u'phase': 0.25,
                                                                                                u'tex': u'sqr',
                                                                                                u'units': u'deg',
                                                                                                u'sf': 0.04,
                                                                                                u'size': [200, 150]},
                                                                                                u'class': u'grating',
                                                                                                u'groups': {u'horizontal': {u'Ori': [90, 270]},
                                                                                                                            u'vertical': {u'Ori': [0, 180]}}},
                                                                      u'failure_repeats': 5,
                                                                      u'warm_up_trials': 5,
                                                                      u'pre_change_time': 2.25},
                                                      u'script': u'http://stash.corp.alleninstitute.org/projects/VB/repos/visual_behavior_scripts/raw/change_detection_with_fingerprint.py?at=021ec55fbbdbb05aad1681c016e83066fe5aa1dd', 'behavior_session_uuid': u'394a910e-94c7-4472-9838-5345aff59ed8', u'script_md5': u'e0535f3b6f03ccc8eeccaed2118f3c1d',
                                                      u'LabTracks_ID': 431151,
                                                      u'date':
                                                      u'2019-02-15T13:01:23.672000',
                                                      'regimen_name': u'VisualBehavior_Task1A_v1.0.1',
                                                      u'default_x': False,
                                                      u'regimens': [{u'active': False, 
                                                                     u'default': False, 
                                                                     u'id': 14, 
                                                                     u'name': u'VisualBehavior_Task1A_v1.0.1'}],
                                                      u'default_y': False}



# def test_get_ophys_experiment_dir(ophys_experiment_id, compare_val):

#     api = LimsOphysAPI()

#     if compare_val is None:
#         expected_fail = False
#         try:
#             api.get_ophys_experiment_dir(ophys_experiment_id)
#         except OneResultExpectedError:
#             expected_fail = True
#         assert expected_fail == True

#     else:
#         api.get_ophys_experiment_dir(ophys_experiment_id=ophys_experiment_id)
#         assert api.get_ophys_experiment_dir(ophys_experiment_id=ophys_experiment_id) == compare_val
