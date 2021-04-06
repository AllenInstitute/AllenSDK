import datetime
import json

import pytz
from pynwb import NWBFile

from allensdk.brain_observatory.behavior2.NWBIO import NWBWriter
from allensdk.internal.api import db_connection_creator
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
import allensdk.brain_observatory.behavior2.data_objects.behavior_session as bs


def test_json_roundtrip(bsession: bs.BehaviorSession):
    """Test a session created from_lims is same as session
    created from_json"""
    dict_repr = bsession.to_json()
    print(json.dumps(dict_repr, indent=2))
    bsession2 = bs.BehaviorSession.from_json(dict_repr)
    dict_repr2 = bsession2.to_json()
    print(json.dumps(dict_repr2, indent=2))
    assert dict_repr2 == dict_repr


def test_nwb_roundtrip(bsession: bs.BehaviorSession):
    """Tests a session created from_lims is same as session created from_nwb"""
    nwbfile = NWBFile(
        session_description='test',
        identifier=str(bsession.behavior_session_id.value),
        session_start_time=pytz.utc.localize(datetime.datetime.now()),
        file_create_date=pytz.utc.localize(datetime.datetime.now()),
        institution="Allen Institute for Brain Science",
        keywords=["visual", "behavior", "task"],
        experiment_description='test'
    )
    bsession.to_nwb(nwbfile=nwbfile)
    nwb_writer = NWBWriter(nwbfile=nwbfile, path='/tmp/test_nwb.nwb')
    nwb_writer.write()

    bsession3 = bs.BehaviorSession.from_nwb_path(path='/tmp/test_nwb.nwb')
    dict_repr = bsession3.to_json()
    print(json.dumps(dict_repr, indent=2))
    assert dict_repr == bsession.to_json()

    
def main():
    dbconn = db_connection_creator(fallback_credentials=LIMS_DB_CREDENTIAL_MAP)
    behavior_session_id = 870987812
    bsession = bs.BehaviorSession.from_lims(
        dbconn, behavior_session_id=behavior_session_id)

    test_json_roundtrip(bsession=bsession)
    test_nwb_roundtrip(bsession=bsession)


if __name__ == '__main__':
    main()
