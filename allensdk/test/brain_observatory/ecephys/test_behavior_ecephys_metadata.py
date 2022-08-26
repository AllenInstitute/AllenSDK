import datetime

import pytest
import pytz
from pynwb import NWBFile

from allensdk.brain_observatory.ecephys._behavior_ecephys_metadata import \
    BehaviorEcephysMetadata


@pytest.fixture(scope='module')
def behavior_ecephys_metadata_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return a BehaviorEcephysMetadata object
    """
    obj = BehaviorEcephysMetadata.from_json(
        dict_repr=behavior_ecephys_session_config_fixture)
    return obj


@pytest.fixture(scope='module')
def ecephys_session_id_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return an ecephys_session_id object
    """
    return behavior_ecephys_session_config_fixture['ecephys_session_id']


def create_nwb_file(
        ecephys_session_id):
    """
    Return an NWB file with a specified ID
    """
    nwbfile = NWBFile(
        session_description='foo',
        identifier=str(ecephys_session_id),
        session_id='foo',
        session_start_time=datetime.datetime(2021, 6, 24, 13, 59, 17, 563000,
                                             tzinfo=pytz.UTC),
        institution="Allen Institute"
    )
    return nwbfile


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip', [True, False])
def test_read_write_nwb(
        roundtrip,
        data_object_roundtrip_fixture,
        behavior_ecephys_metadata_fixture,
        ecephys_session_id_fixture):

    nwbfile = create_nwb_file(ecephys_session_id_fixture)
    behavior_ecephys_metadata_fixture.to_nwb(nwbfile=nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=BehaviorEcephysMetadata)
    else:
        obt = BehaviorEcephysMetadata.from_nwb(nwbfile=nwbfile)

    assert obt == behavior_ecephys_metadata_fixture
