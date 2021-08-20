import pytest

from allensdk.brain_observatory.behavior.mtrain import ExtendedTrialSchema
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsApi)
from marshmallow.schema import ValidationError


@pytest.mark.requires_bamboo
@pytest.mark.parametrize(
    'behavior_experiment_id', [
        880293569,  # stage: TRAINING_0_gratings_autorewards_15min
        881236782,  # stage: TRAINING_1_gratings
        881236761,  # stage: TRAINING_2_gratings_flashed
    ]
)
def test_get_extended_trials(behavior_experiment_id):
    api = BehaviorLimsApi(behavior_experiment_id)
    df = api.get_extended_trials()
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df.to_dict('records')
    data_list_cs_sc = ets.dump(data_list_cs)
    ets.load(data_list_cs_sc)

    df_fail = df.drop(['behavior_session_uuid'], axis=1)
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df_fail.to_dict('records')
    data_list_cs_sc = ets.dump(data_list_cs)
    try:
        ets.load(data_list_cs_sc)
        raise RuntimeError("This should have failed with "
                           "marshmallow.schema.ValidationError")
    except ValidationError:
        pass
