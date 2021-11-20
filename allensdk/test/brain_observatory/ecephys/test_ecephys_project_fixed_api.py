import pytest

from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectFixedApi, MissingDataError


def test_get_sessions():
    api = EcephysProjectFixedApi()
    with pytest.raises(MissingDataError) as err:
        api.get_sessions()


def test_get_session_data():
    api = EcephysProjectFixedApi()
    with pytest.raises(MissingDataError) as err:
        api.get_session_data(12345)
        assert re.compile("12345").search(err.message) is not None