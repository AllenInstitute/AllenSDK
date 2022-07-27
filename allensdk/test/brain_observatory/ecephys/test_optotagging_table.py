import pytest

from allensdk.brain_observatory.ecephys.optotagging import OptotaggingTable


@pytest.fixture(scope='module')
def optotagging_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return an OptotaggingTable
    """
    obj = OptotaggingTable.from_json(
        dict_repr=behavior_ecephys_session_config_fixture)
    return obj


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip', [True, False])
def test_read_write_nwb(roundtrip,
                        data_object_roundtrip_fixture,
                        optotagging_fixture,
                        helper_functions):
    nwbfile = helper_functions.create_blank_nwb_file()
    optotagging_fixture.to_nwb(nwbfile=nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=OptotaggingTable)
    else:
        obt = OptotaggingTable.from_nwb(nwbfile=nwbfile)

    assert obt == optotagging_fixture
