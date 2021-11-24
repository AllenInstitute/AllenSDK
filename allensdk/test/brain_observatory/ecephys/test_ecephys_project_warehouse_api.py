import pytest

from allensdk.brain_observatory.ecephys.ecephys_project_api import ecephys_project_warehouse_api as epwa

@pytest.mark.skipif(True, reason="broken test")
@pytest.mark.parametrize(
    "method,conditions,expected_query",
    [
        [
            "get_sessions",
            {},
            (
                "criteria=model::EcephysSession"
            ),
        ],
        [
            "get_sessions",
            {"session_ids": [779839471, 759228117]},
            (
                "criteria=model::EcephysSession,rma::criteria[id$in779839471,759228117]"
            ),
        ],
        [
            "get_sessions",
            {"session_ids": [779839471, 759228117], "has_eye_tracking": True},
            (
                "criteria=model::EcephysSession,rma::criteria[id$in779839471,759228117][fail_eye_tracking$eqfalse]"
            ),
        ],
        [
            "get_sessions",
            {"session_ids": [779839471, 759228117], "has_eye_tracking": True, "stimulus_names": ["foo", "bar"]},
            (
                "criteria=model::EcephysSession,rma::criteria[id$in779839471,759228117][fail_eye_tracking$eqfalse][stimulus_name$in'foo','bar']"
            ),
        ],
        [
            "get_probes",
            {"session_ids": [797828357, 774875821], "probe_ids": [805579741, 792602660]},
            (
                "criteria=model::EcephysProbe,rma::criteria[id$in805579741,792602660][ecephys_session_id$in797828357,774875821]"
            ),
        ],
        [
            "get_channels",
            {"session_ids": [746083955], "probe_ids": [760647913], "channel_ids": [849734900]},
            (
                "criteria=model::EcephysChannel,rma::criteria[id$in849734900][ecephys_probe_id$in760647913],rma::criteria,ecephys_probe[ecephys_session_id$in746083955]"
            ),
        ],
        [
            "get_units",
            {"session_ids": [779839471], "probe_ids": [792645497], "channel_ids": [849709694], "unit_ids": [849710462]},
            (
                "criteria=model::EcephysUnit,"
                "rma::criteria[id$in849710462],"
                "rma::criteria[ecephys_channel_id$in849709694],"
                "rma::criteria,ecephys_channel(ecephys_probe[id$in792645497]),"
                "rma::criteria,ecephys_channel(ecephys_probe(ecephys_session[id$in779839471]))"
            ),
        ],
    ],
)
def test_query(method, conditions, expected_query):
    class MockRmaEngine:
        def get_rma_tabular(self, rendered):
            print(expected_query)
            print(rendered)
            assert expected_query == rendered
            return []

    api = epwa.EcephysProjectWarehouseApi(rma_engine=MockRmaEngine())
    results = getattr(api, method)(**conditions)
