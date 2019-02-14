import pytest

from allensdk.internal.api import OneResultExpectedError
from allensdk.internal.api.mtrain_api import MtrainApi

@pytest.mark.nightly
def test_get_subjects():

    api = MtrainApi()
    subject_list = api.get_subjects()
    assert len(subject_list) > 190 and 423746 in subject_list

@pytest.mark.nightly
@pytest.mark.parametrize('LabTracks_ID', [
    pytest.param(423986),
])
def test_get_behavior_training_df(LabTracks_ID):
    
    api = MtrainApi()
    df = api.get_behavior_training_df(423986)
    assert list(df.columns) == [u'stage_name', u'regimen_name', u'date', u'behavior_session_id']
    assert len(df) == 24
    # raise


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

