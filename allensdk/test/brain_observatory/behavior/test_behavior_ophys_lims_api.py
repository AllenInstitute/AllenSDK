import pytest

from allensdk.internal.api import OneResultExpectedError
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi


@pytest.mark.nightly
@pytest.mark.parametrize('ophys_experiment_id, compare_val', [
    pytest.param(789359614, '/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl'),
    pytest.param(0, None)
])
def test_get_behavior_stimulus_file(ophys_experiment_id, compare_val):

    api = BehaviorOphysLimsApi()

    if compare_val is None:
        expected_fail = False
        try:
            api.get_behavior_stimulus_file(ophys_experiment_id)
        except OneResultExpectedError:
            expected_fail = True
        assert expected_fail is True
    else:
        assert api.get_behavior_stimulus_file(
            ophys_experiment_id=ophys_experiment_id) == compare_val
