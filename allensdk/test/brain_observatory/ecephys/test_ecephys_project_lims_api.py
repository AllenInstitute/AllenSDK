import os
import re

import pytest
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.ecephys_project_api import (
    ecephys_project_lims_api as epla,
)


@pytest.mark.parametrize(
    "method,conditions,expected",
    [
        [
            "get_sessions",
            {"published": True},
            re.compile(r".*where true and es\.workflow_state in \('uploaded'\) and es\.habituation = false and es\.published_at is not null and pr\.name in \('BrainTV Neuropixels Visual Behavior','BrainTV Neuropixels Visual Coding'\)$"),
        ],
        [
            "get_sessions",
            {"session_ids": [1, 2, 3]},
            re.compile(r".*and es\.id in \(1,2,3\).*"),
        ],
        [
            "get_units",
            {"session_ids": [1, 2, 3]},
            re.compile(r"select eu\.\*.*and es\.id in \(1,2,3\) and eu.amplitude_cutoff <= 0.1 and eu.presence_ratio >= 0.95 and eu.isi_violations <= 0.5$"),
        ],
        [
            "get_units",
            {"session_ids": [1, 2, 3], "unit_ids": (4, 5, 6)},
            re.compile(r"select eu\.\*.*and eu\.id in \(4,5,6\) and es\.id in \(1,2,3\) and eu.amplitude_cutoff <= 0.1 and eu.presence_ratio >= 0.95 and eu.isi_violations <= 0.5$")
        ],
        [
            "get_channels",
            {},
            re.compile(r"select ec\.id as id.*where valid_data and ep.workflow_state != 'failed' and es.workflow_state != 'failed'$"),
        ],
        [
            "get_probes",
            {},
            re.compile(r"select ep\.id as id, ep.ecephys_session_id.*where true and ep.workflow_state != 'failed' and es.workflow_state != 'failed'$"),
        ],
    ],
)
def test_query(method, conditions, expected):
    class MockPgEngine:
        def select(self, rendered):
            self.query = " ".join([item.strip() for item in str(rendered).split()])
            return pd.DataFrame({"id": [1, 2, 3], "ecephys_channel_id": [1, 2, 3], "genotype": [np.nan, "a", "b"]})

    pg_engine = MockPgEngine()
    api = epla.EcephysProjectLimsApi(postgres_engine=pg_engine, app_engine=None)

    results = getattr(api, method)(**conditions)
    
    obtained = pg_engine.query.strip()
    print(obtained)
    match = expected.match(obtained)
    assert match is not None


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
                r".*and earp.ecephys_probe_id = (?P<probe_id>\d+).*", re.DOTALL
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
