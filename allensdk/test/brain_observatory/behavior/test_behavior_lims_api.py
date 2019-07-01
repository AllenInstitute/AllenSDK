import pytest

from allensdk import OneResultExpectedError
from allensdk.internal.api.behavior_lims_api import BehaviorLimsApi
from allensdk.brain_observatory.behavior.mtrain import ExtendedTrialSchema
from marshmallow.schema import ValidationError


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_experiment_id, compare_val', [
    pytest.param(880293569, '/allen/programs/braintv/production/neuralcoding/prod0/specimen_703198163/behavior_session_880293569/880289456.pkl'),
    pytest.param(0, None)
])
def test_get_behavior_stimulus_file(behavior_experiment_id, compare_val):
    api = BehaviorLimsApi(behavior_experiment_id)

    if compare_val is None:
        expected_fail = False
        try:
            api.get_behavior_stimulus_file()
        except OneResultExpectedError:
            expected_fail = True
        assert expected_fail is True
    else:
        assert api.get_behavior_stimulus_file() == compare_val


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_session_uuid', [
    pytest.param('394a910e-94c7-4472-9838-5345aff59ed8'),
])
def test_foraging_id_to_behavior_session_id(behavior_session_uuid):
    behavior_session_id = BehaviorLimsApi.foraging_id_to_behavior_session_id(
        behavior_session_uuid)
    assert behavior_session_id == 823847007


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_session_uuid', [
    pytest.param(823847007),
])
def test_behavior_session_id_to_foraging_id(behavior_session_uuid):
    foraging_id = BehaviorLimsApi.behavior_session_id_to_foraging_id(
        behavior_session_uuid)
    assert foraging_id == '394a910e-94c7-4472-9838-5345aff59ed8'


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
        raise RuntimeError('This should have failed with marshmallow.schema.ValidationError')
    except ValidationError:
        pass
