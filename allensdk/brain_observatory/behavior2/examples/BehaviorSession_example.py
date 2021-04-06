import json
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


if __name__ == '__main__':
    main()
