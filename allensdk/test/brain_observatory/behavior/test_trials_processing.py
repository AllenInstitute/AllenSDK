import pytest
import pandas as pd

from allensdk.internal.api.behavior_lims_api import BehaviorLimsApi
from allensdk.brain_observatory.behavior import trials_processing


@pytest.mark.requires_bamboo
@pytest.mark.parametrize(
    'behavior_experiment_id, ti, expected, exception', [
        (880293569, 5, (90, 90, None), None, ),
        (881236761, 0, None, IndexError, )
    ]
)
def test_get_ori_info_from_trial(behavior_experiment_id, ti, expected, exception, ):
    """was feeling worried that the values would be wrong,
    this helps reaffirm that maybe they are not...

    Notes
    -----
    - i may be rewriting code here but its more a sanity check really...
    """
    stim_output = pd.read_pickle(
        BehaviorLimsApi(behavior_experiment_id).get_behavior_stimulus_file()
    )
    trial_log = stim_output['items']['behavior']['trial_log']

    if exception:
        with pytest.raises(exception):
            trials_processing.get_ori_info_from_trial(trial_log, ti, )
    else:
        assert trials_processing.get_ori_info_from_trial(trial_log, ti, ) == expected