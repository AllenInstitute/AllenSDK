import os
import re

import pytest
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_api import (
    ecephys_project_lims_api as epla,
)


@pytest.mark.parametrize(
    "method,conditions,expected_query",
    [
        [
            "get_sessions",
            {"published": True},
            (
                "select es.* from ecephys_sessions es "
                "join projects pr on pr.id = es.project_id "
                "where true "
                "and es.workflow_state in ('uploaded') "
                "and es.habituation = false "
                "and es.published_at is not null "
                "and pr.name in ('BrainTV Neuropixels Visual Behavior','BrainTV Neuropixels Visual Coding')"
            ),
        ],
        [
            "get_sessions",
            {"session_ids": [1, 2, 3]},
            (
                "select es.* from ecephys_sessions es "
                "join projects pr on pr.id = es.project_id "
                "where true "
                "and es.id in (1,2,3) "
                "and es.workflow_state in ('uploaded') "
                "and es.habituation = false "
                "and pr.name in ('BrainTV Neuropixels Visual Behavior','BrainTV Neuropixels Visual Coding')"
            ),
        ],
        [
            "get_units",
            {"session_ids": [1, 2, 3]},
            (
                "select eu.* from ecephys_units eu "
                "join ecephys_channels ec on ec.id = eu.ecephys_channel_id "
                "join ecephys_probes ep on ep.id = ec.ecephys_probe_id "
                "join ecephys_sessions es on es.id = ep.ecephys_session_id "
                "where true "
                "and es.id in (1,2,3)"
            ),
        ],
        [
            "get_units",
            {"session_ids": [1, 2, 3], "unit_ids": (4, 5, 6)},
            (
                "select eu.* from ecephys_units eu "
                "join ecephys_channels ec on ec.id = eu.ecephys_channel_id "
                "join ecephys_probes ep on ep.id = ec.ecephys_probe_id "
                "join ecephys_sessions es on es.id = ep.ecephys_session_id "
                "where true "
                "and eu.id in (4,5,6) "
                "and es.id in (1,2,3)"
            ),
        ],
        [
            "get_channels",
            {},
            (
                "select ec.* from ecephys_channels ec "
                "join ecephys_probes ep on ep.id = ec.ecephys_probe_id "
                "join ecephys_sessions es on es.id = ep.ecephys_session_id "
                "where true "
            ),
        ],
        [
            "get_probes",
            {},
            (
                "select ep.* from ecephys_probes ep "
                "join ecephys_sessions es on es.id = ep.ecephys_session_id "
                "where true "
            ),
        ],
    ],
)
def test_query(method, conditions, expected_query):
    class MockPgEngine:
        def select(self, rendered):
            self.query = " ".join([item.strip() for item in str(rendered).split()])
            return pd.DataFrame({})

    pg_engine = MockPgEngine()
    api = epla.EcephysProjectLimsApi(postgres_engine=pg_engine, app_engine=None)

    results = getattr(api, method)(**conditions)
    assert expected_query.strip() == pg_engine.query.strip()


def test_get_session_data():

    session_id = 12345
    wkf_id = 987

    class MockPgEngine:
        def select(self, rendered):
            pattern = re.compile(
                r".*and ear.ecephys_session_id = (?P<session_id>\d+).*", re.DOTALL
            )
            match = pattern.match(rendered)
            sid_obt = int(match["session_id"])
            assert session_id == sid_obt
            return pd.DataFrame({"id": [wkf_id]})

    class MockHttpEngine:
        def stream(self, path):
            assert path == f"well_known_files/download/{wkf_id}?wkf_id={wkf_id}"

    api = epla.EcephysProjectLimsApi(
        postgres_engine=MockPgEngine(), app_engine=MockHttpEngine()
    )
    api.get_session_data(session_id)


def test_get_probe_data():

    probe_id = 12345
    wkf_id = 987

    class MockPgEngine:
        def select(self, rendered):
            pattern = re.compile(
                r".*and earp.probe_id = (?P<probe_id>\d+).*", re.DOTALL
            )
            match = pattern.match(rendered)
            pid_obt = int(match["probe_id"])
            assert probe_id == pid_obt
            return pd.DataFrame({"id": [wkf_id]})

    class MockHttpEngine:
        def stream(self, path):
            assert path == f"well_known_files/download/{wkf_id}?wkf_id={wkf_id}"

    api = epla.EcephysProjectLimsApi(
        postgres_engine=MockPgEngine(), app_engine=MockHttpEngine()
    )
    api.get_probe_lfp_data(probe_id)
