import os

import pytest

import allensdk.brain_observatory.ecephys.ecephys_project_cache as epc
from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectLimsApi

@pytest.fixture
def mock_api(sessions):
    class MockApi:
        @cacheable()
        def get_sessions(self):



def test_get_sessions(tmpdir_factory, mock_api, sessions):

    tmpdir = str(tmpdir_factory.mktemp('test_ecephys_project_cache'))
    man_path = os.path.join(tmpdir, 'manifest.json')

