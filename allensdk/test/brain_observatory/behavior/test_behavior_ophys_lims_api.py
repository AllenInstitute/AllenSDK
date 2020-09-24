import pytest
import pandas as pd
import numpy as np

from allensdk.internal.api import OneResultExpectedError
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.mtrain import ExtendedTrialSchema
from marshmallow.schema import ValidationError


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('ophys_experiment_id, compare_val', [
    pytest.param(789359614, '/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl'),
    pytest.param(0, None)
])
def test_get_behavior_stimulus_file(ophys_experiment_id, compare_val):

    api = BehaviorOphysLimsApi(ophys_experiment_id)

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
@pytest.mark.parametrize('ophys_experiment_id', [789359614])
def test_get_extended_trials(ophys_experiment_id):

    api = BehaviorOphysLimsApi(ophys_experiment_id)
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


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('ophys_experiment_id', [860030092])
def test_get_nwb_filepath(ophys_experiment_id):

    api = BehaviorOphysLimsApi(ophys_experiment_id)
    assert api.get_nwb_filepath() == '/allen/programs/braintv/production/visualbehavior/prod0/specimen_823826986/ophys_session_859701393/ophys_experiment_860030092/behavior_ophys_session_860030092.nwb'


@pytest.mark.parametrize(
    "timestamps,plane_group,group_count,expected",
    [
        (np.ones(10), 1, 0, np.ones(10)),
        (np.ones(10), 1, 0, np.ones(10)),
        # middle
        (np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 1, 3, np.ones(4)),
        # first
        (np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 0, 4, np.ones(3)),
        # last
        (np.array([0, 1, 0, 1, 0, 1, 0, 1]), 1, 2, np.ones(4)),
        # only one group
        (np.ones(10), 0, 1, np.ones(10))
    ]
)
def test_process_ophys_plane_timestamps(
        timestamps, plane_group, group_count, expected):
    actual = BehaviorOphysLimsApi._process_ophys_plane_timestamps(
        timestamps, plane_group, group_count)
    np.testing.assert_array_equal(expected, actual)
