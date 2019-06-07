import pytest

from allensdk.brain_observatory.ecephys.ecephys_project_api import ecephys_project_warehouse_api as epwa

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
    ],
)
def test_query(method, conditions, expected_query):
    class MockRmaEngine:
        def get_rma_tabular(self, rendered):
            assert expected_query == rendered
            return []

    api = epwa.EcephysProjectWarehouseApi(rma_engine=MockRmaEngine())
    results = getattr(api, method)(**conditions)
