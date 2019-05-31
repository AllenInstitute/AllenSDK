import pytest

from allensdk.internal.api import OneResultExpectedError
from allensdk.internal.api.behavior_lims_api import BehaviorLimsApi


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_session_uuid', [
    pytest.param('394a910e-94c7-4472-9838-5345aff59ed8'),
])
def test_foraging_id_to_behavior_session_id(behavior_session_uuid):
    api = BehaviorLimsApi()
    behavior_session_id = api.foraging_id_to_behavior_session_id(
        behavior_session_uuid)
    assert behavior_session_id == 823847007


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_session_uuid', [
    pytest.param(823847007),
])
def test_behavior_session_id_to_foraging_id(behavior_session_uuid):
    api = BehaviorLimsApi()
    foraging_id = api.behavior_session_id_to_foraging_id(
        behavior_session_uuid)
    assert foraging_id == '394a910e-94c7-4472-9838-5345aff59ed8'
