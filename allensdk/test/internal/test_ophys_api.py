import pytest

from allensdk.internal.api import OneResultExpectedError
from allensdk.internal.api.lims_ophys_api import LimsOphysAPI


@pytest.mark.parametrize('ophys_experiment_id, compare_val', [
    pytest.param(702134928, '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/'),
    pytest.param(0, None)
])
def test_get_ophys_experiment_dir(ophys_experiment_id, compare_val):

    api = LimsOphysAPI()

    if compare_val is None:
        expected_fail = False
        try:
            api.get_ophys_experiment_dir(ophys_experiment_id)
        except OneResultExpectedError:
            expected_fail = True
        assert expected_fail == True
    
    else:
        api.get_ophys_experiment_dir(ophys_experiment_id=ophys_experiment_id)
        assert api.get_ophys_experiment_dir(ophys_experiment_id=ophys_experiment_id) == compare_val

