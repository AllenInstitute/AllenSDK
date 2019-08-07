import os
import collections

import pytest
import pandas as pd
import mock

import allensdk.brain_observatory.ecephys.ecephys_project_cache as epc


@pytest.fixture
def sessions():
    return pd.DataFrame({
        'stimulus_name': ['stimulus_set_one', 'stimulus_set_two', 'stimulus_set_two'],
        "unit_count": [500, 1000, 1500],
        "channel_count": [40, 90, 140],
        "probe_count": [3, 4, 5],
        "structure_acronyms": [["a", "v"], ["a", "c"], ["b"]]
    }, index=pd.Series(name='id', data=[1, 2, 3]))


@pytest.fixture
def units():
    return pd.DataFrame({
        'peak_channel_id': [2, 1],
        'snr': [1.5, 4.9]
    }, index=pd.Series(name='id', data=[1, 2]))


@pytest.fixture
def channels():
    return pd.DataFrame({
        'ecephys_probe_id': [11, 11],
        'ap': [1000, 2000],
        "unit_count": [5, 10]
    }, index=pd.Series(name='id', data=[1, 2]))


@pytest.fixture
def probes():
    return pd.DataFrame({
        'ecephys_session_id': [3],
        "unit_count": [50],
        "channel_count": [10]
    }, index=pd.Series(name='id', data=[11]))


@pytest.fixture
def shared_tmpdir(tmpdir_factory):
    return str(tmpdir_factory.mktemp('test_ecephys_project_cache'))


@pytest.fixture
def mock_api(shared_tmpdir, sessions, units, channels, probes):
    class MockApi:
        
        def __init__(self, **kwargs):
            self.accesses = collections.defaultdict(lambda: 1)

        def __getattr__(self, name):
            self.accesses[name] += 1

        def get_sessions(self):
            return sessions

        def get_units(self):
            return units

        def get_channels(self):
            return channels

        def get_probes(self):
            return probes

        def get_session_data(self, session_id):
            path = os.path.join(shared_tmpdir, 'tmp.txt')
            with open(path, 'w') as f:
                f.write(f'{session_id}')
            return open(path, 'rb')

    return MockApi


@pytest.fixture
def tmpdir_cache(shared_tmpdir, mock_api):

    man_path = os.path.join(shared_tmpdir, 'manifest.json')

    return epc.EcephysProjectCache(
        fetch_api=mock_api(),
        manifest=man_path
    )


def lazy_cache_test(cache, cache_name, api_name, expected):
    obtained_one = getattr(cache, cache_name)()
    obtained_two = getattr(cache, cache_name)()

    pd.testing.assert_frame_equal(expected, obtained_one)
    pd.testing.assert_frame_equal(expected, obtained_two)

    assert 1 == cache.fetch_api.accesses[api_name]


def test_get_sessions(tmpdir_cache, sessions):
    lazy_cache_test(tmpdir_cache, 'get_sessions', "get_sessions", sessions)


def test_get_units(tmpdir_cache, units):
    lazy_cache_test(tmpdir_cache, 'get_units', "get_units", units)


def test_get_units_annotated(tmpdir_cache, units, channels, probes, sessions):
    units = tmpdir_cache.get_units(annotate=True)
    assert units.loc[2, "stimulus_name"] == "stimulus_set_two"


def test_get_probes(tmpdir_cache, probes):
    lazy_cache_test(tmpdir_cache, 'get_probes', "get_probes", probes)


def test_get_channels(tmpdir_cache, channels):
    lazy_cache_test(tmpdir_cache, 'get_channels', "get_channels", channels)


def test_get_session_data(shared_tmpdir, tmpdir_cache):

    sid = 12345

    data_one = tmpdir_cache.get_session_data(sid)
    data_two = tmpdir_cache.get_session_data(sid)

    assert 1 == tmpdir_cache.fetch_api.accesses['get_session_data']
    assert os.path.join(shared_tmpdir, f"session_{sid}", f"session_{sid}.nwb") == data_one.api.path