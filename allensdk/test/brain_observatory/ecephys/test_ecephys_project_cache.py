import os
import collections

import pytest
import pandas as pd
import numpy as np
import SimpleITK as sitk

import allensdk.brain_observatory.ecephys.ecephys_project_cache as epc


@pytest.fixture
def raw_sessions():
    return pd.DataFrame({
        'session_type': ['stimulus_set_one', 'stimulus_set_two', 'stimulus_set_two'],
        "unit_count": [500, 1000, 1500],
        "channel_count": [40, 90, 140],
        "probe_count": [3, 4, 5],
        "structure_acronyms": [["a", "v"], ["a", "c"], ["b"]]
    }, index=pd.Series(name='id', data=[1, 2, 3]))


@pytest.fixture
def sessions():
    return pd.DataFrame({
        'session_type': ['stimulus_set_one', 'stimulus_set_two', 'stimulus_set_two'],
        "unit_count": [500, 1000, 1500],
        "channel_count": [40, 90, 140],
        "probe_count": [3, 4, 5],
        "ecephys_structure_acronyms": [["a", "v"], ["a", "c"], ["b"]]
    }, index=pd.Series(name='id', data=[1, 2, 3]))


@pytest.fixture
def units():
    return pd.DataFrame({
        'ecephys_channel_id': [2, 1],
        'snr': [1.5, 4.9],
        "amplitude_cutoff": [0.05, 0.2],
        "presence_ratio": [10, 20],
        "isi_violations": [0.3, 0.4]
    }, index=pd.Series(name='id', data=[1, 2]))


@pytest.fixture
def filtered_units():
    return pd.DataFrame({
        'ecephys_channel_id': [3],
        'snr': [4.2],
        'amplitude_cutoff': [0.08],
        'presence_ratio': [15],
        'isi_violations': [0.35]
    }, index=pd.Series(name='id', data=[3]))


@pytest.fixture
def analysis_metrics():
    return pd.DataFrame({
        "a": [0, 1, 2],
        "b": [3, 4, 5]
    }, index=pd.Index(name="ecephys_unit_id", data=[1, 2, 3]))


@pytest.fixture
def channels():
    return pd.DataFrame({
        'ecephys_probe_id': [11, 11],
        'ap': [1000, 2000],
        "unit_count": [5, 10],
        "ecephys_structure_acronym": ["a", "b"]
    }, index=pd.Series(name='id', data=[1, 2]))


@pytest.fixture
def raw_probes():
    return pd.DataFrame({
        'ecephys_session_id': [3],
        "unit_count": [50],
        "channel_count": [10],
        "lfp_temporal_subsampling_factor": [2.0],
        "lfp_sampling_rate": [1000.0],
    }, index=pd.Series(name='id', data=[11]))


@pytest.fixture
def probes():
    return pd.DataFrame({
        'ecephys_session_id': [3],
        "unit_count": [50],
        "channel_count": [10],
        "lfp_temporal_subsampling_factor": [2.0],
        "lfp_sampling_rate": [500.0],
    }, index=pd.Series(name='id', data=[11]))


@pytest.fixture
def annotated_probes(probes, sessions):
    return pd.merge(probes, sessions, left_on="ecephys_session_id", right_index=True, suffixes=["_probe", "_session"])


@pytest.fixture
def annotated_channels(channels, annotated_probes):
    return pd.merge(channels, annotated_probes, left_on="ecephys_probe_id", right_index=True, suffixes=["_channel", "_probe"])


@pytest.fixture
def annotated_units(units, annotated_channels):
    return pd.merge(units, annotated_channels, left_on="ecephys_channel_id", right_index=True, suffixes=["_unit", "_channel"])


@pytest.fixture
def shared_tmpdir(tmpdir_factory):
    return str(tmpdir_factory.mktemp('test_ecephys_project_cache'))


@pytest.fixture
def mock_api(shared_tmpdir, raw_sessions, units, filtered_units, channels, raw_probes, analysis_metrics):
    class MockApi:

        def __init__(self, **kwargs):
            self.accesses = collections.defaultdict(lambda: 1)

        def __getattr__(self, name):
            self.accesses[name] += 1

        def get_sessions(self, **kwargs):
            return raw_sessions

        def get_units(self, **kwargs):
            if kwargs['filter_by_validity']:
                return filtered_units
            else:
                return units

        def get_channels(self, **kwargs):
            return channels

        def get_probes(self, **kwargs):
            return raw_probes

        def get_session_data(self, session_id, **kwargs):
            assert kwargs.get('filter_by_validity')
            path = os.path.join(shared_tmpdir, 'tmp.txt')
            with open(path, 'w') as f:
                f.write(f'{session_id}')
            return open(path, 'rb')

        def get_natural_scene_template(self, number):
            path = os.path.join(shared_tmpdir, "tmp.tiff")
            img = sitk.GetImageFromArray(np.eye(100, dtype=np.uint8))
            sitk.WriteImage(img, path)
            return open(path, "rb")

        def get_natural_movie_template(self, number):
            path = os.path.join(shared_tmpdir, "tmp.npy")
            np.save(path, np.eye(100))
            return open(path, "rb")

        def get_unit_analysis_metrics(self, *a, **k):
            return analysis_metrics

    return MockApi


@pytest.fixture
def tmpdir_cache(shared_tmpdir, mock_api):

    man_path = os.path.join(shared_tmpdir, 'manifest.json')

    return epc.EcephysProjectCache(
        fetch_api=mock_api(),
        manifest=man_path
    )


def lazy_cache_test(cache, cache_name, api_name, expected, *args, **kwargs):
    obtained_one = getattr(cache, cache_name)(*args, **kwargs)
    obtained_two = getattr(cache, cache_name)(*args, **kwargs)

    pd.testing.assert_frame_equal(expected, obtained_one)
    pd.testing.assert_frame_equal(expected, obtained_two)

    assert 1 == cache.fetch_api.accesses[api_name]


def test_get_sessions(tmpdir_cache, sessions):
    lazy_cache_test(tmpdir_cache, '_get_sessions', "get_sessions", sessions)


@pytest.mark.parametrize("filter_by_validity", [False, True])
def test_get_units(tmpdir_cache, units, filtered_units, filter_by_validity):
    if filter_by_validity:
        lazy_cache_test(tmpdir_cache, '_get_units', "get_units", filtered_units, filter_by_validity=filter_by_validity)
    else:
        units = units[units["amplitude_cutoff"] <= 0.1]
        lazy_cache_test(tmpdir_cache, '_get_units', "get_units", units, filter_by_validity=filter_by_validity)


def test_get_probes(tmpdir_cache, probes):
    lazy_cache_test(tmpdir_cache, '_get_probes', "get_probes", probes)


def test_get_channels(tmpdir_cache, channels):
    lazy_cache_test(tmpdir_cache, '_get_channels', "get_channels", channels)


def test_get_annotated_probes(tmpdir_cache, probes, annotated_probes):
    lazy_cache_test(tmpdir_cache, "_get_annotated_probes", "get_probes", annotated_probes)


def test_get_annotated_channels(tmpdir_cache, channels, annotated_channels):
    lazy_cache_test(tmpdir_cache, "_get_annotated_channels", "get_channels", annotated_channels)


def test_get_annotated_units(tmpdir_cache, units, annotated_units):
    annotated_units = annotated_units[annotated_units["amplitude_cutoff"] < 0.1]

    lazy_cache_test(tmpdir_cache, "_get_annotated_units", "get_units", annotated_units, filter_by_validity=False)


def test_get_session_data(shared_tmpdir, tmpdir_cache):

    sid = 12345

    data_one = tmpdir_cache.get_session_data(sid)

    assert 1 == tmpdir_cache.fetch_api.accesses['get_session_data']
    assert os.path.join(shared_tmpdir, f"session_{sid}", f"session_{sid}.nwb") == data_one.api.path


def test_get_natural_scene_template(shared_tmpdir, tmpdir_cache):
    num = 10

    data_one = tmpdir_cache.get_natural_scene_template(num)

    assert 1 == tmpdir_cache.fetch_api.accesses["get_natural_scene_template"]
    assert np.allclose(np.eye(100), data_one)


def test_get_natural_movie_template(shared_tmpdir, tmpdir_cache):
    num = 10

    data_one = tmpdir_cache.get_natural_movie_template(num)

    assert 1 == tmpdir_cache.fetch_api.accesses["get_natural_movie_template"]
    assert np.allclose(np.eye(100), data_one)


def test_get_unit_analysis_metrics_for_session(tmpdir_cache, analysis_metrics):
    lazy_cache_test(
        tmpdir_cache,
        'get_unit_analysis_metrics_for_session',
        "get_unit_analysis_metrics",
        analysis_metrics,
        session_id=3,
        annotate=False
    )


def test_get_unit_analysis_metrics_by_session_type(tmpdir_cache, analysis_metrics):
    lazy_cache_test(
        tmpdir_cache,
        'get_unit_analysis_metrics_by_session_type',
        "get_unit_analysis_metrics",
        analysis_metrics,
        session_type="stimulus_set_two",
        annotate=False
    )
