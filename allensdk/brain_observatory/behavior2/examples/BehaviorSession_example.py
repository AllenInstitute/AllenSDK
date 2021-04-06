import datetime
import json

import pytz
from pynwb import NWBFile

from allensdk.internal.api import db_connection_creator
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
import allensdk.brain_observatory.behavior2.data_objects.behavior_session as bs


def main():
    dbconn = db_connection_creator(fallback_credentials=LIMS_DB_CREDENTIAL_MAP)
    eid = 843007058
    bsession = bs.BehaviorSession.from_lims(dbconn, eid)

    # simple round-trip
    dict_repr = bsession.to_json()
    print(json.dumps(dict_repr, indent=2))
    bsession2 = bs.BehaviorSession.from_json(dict_repr)
    dict_repr2 = bsession2.to_json()
    print(json.dumps(dict_repr2, indent=2))
    assert dict_repr2 == dict_repr

    nwbfile = NWBFile(
        session_description='test',
        identifier=str(bsession2.behavior_session_id),
        session_start_time=pytz.utc.localize(datetime.datetime.now()),
        file_create_date=pytz.utc.localize(datetime.datetime.now()),
        institution="Allen Institute for Brain Science",
        keywords=["visual", "behavior", "task"],
        experiment_description='test'
    )
    bsession2.to_nwb(nwbfile=nwbfile)


if __name__ == '__main__':
    main()
