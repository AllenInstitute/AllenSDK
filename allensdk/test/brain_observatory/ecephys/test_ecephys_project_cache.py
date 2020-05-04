import os
import collections
from datetime import datetime

import pytest
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pynwb

import allensdk.brain_observatory.ecephys.ecephys_project_cache as epc
import allensdk.brain_observatory.ecephys.write_nwb.__main__ as write_nwb
from allensdk.brain_observatory.ecephys.ecephys_project_api.http_engine import (
    write_from_stream, write_bytes_from_coroutine, AsyncHttpEngine, HttpEngine
)


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
        "isi_violations": [0.3, 0.4],
        "quality": ["good", "noise"]
    }, index=pd.Series(name='id', data=[1, 2]))


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


class MockEngine:
    def __init__(self):
        self.write_bytes = write_from_stream


@pytest.fixture
def mock_api(shared_tmpdir, raw_sessions, units, channels, raw_probes, analysis_metrics):
    class MockApi:

        def __init__(self, **kwargs):
            self.accesses = collections.defaultdict(lambda: 1)
            self.rma_engine = MockEngine()

        def __getattr__(self, name):
            self.accesses[name] += 1

        def get_sessions(self, **kwargs):
            return raw_sessions

        def get_units(self, **kwargs):
            return units

        def get_channels(self, **kwargs):
            return channels

        def get_probes(self, **kwargs):
            return raw_probes

        def get_session_data(self, session_id, **kwargs):
            path = os.path.join(shared_tmpdir, 'tmp.nwb')

            nwbfile = pynwb.NWBFile(
                session_description='EcephysSession',
                identifier=f"{session_id}",
                session_start_time=datetime.now()
            )

            write_nwb.add_probe_to_nwbfile(nwbfile, 11, sampling_rate=1.0,
                                           lfp_sampling_rate=2.0,
                                           has_lfp_data=True,
                                           name="Test Probe")

            with pynwb.NWBHDF5IO(path, "w") as io:
                io.write(nwbfile)

            return open(path, 'rb')

        def get_probe_lfp_data(self, probe_id):
            path = os.path.join(shared_tmpdir, f"probe_{probe_id}.nwb")

            nwbfile = pynwb.NWBFile(
                session_description='EcephysProbe',
                identifier=f"{probe_id}",
                session_start_time=datetime.now()
            )

            with pynwb.NWBHDF5IO(path, "w") as io:
                io.write(nwbfile)

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
def test_get_units(tmpdir_cache, units, filter_by_validity):
    if filter_by_validity:
        units = units[units["quality"] == "good"].drop(columns="quality")
        lazy_cache_test(tmpdir_cache, '_get_units', "get_units", units, filter_by_validity=filter_by_validity)
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


def test_get_session_data_eventual_success(tmpdir_factory, mock_api):
    man_path = os.path.join(
        tmpdir_factory.mktemp("get_session_data"),
        "manifest.json"
    )

    class InitiallyFailingApi(mock_api):
        def get_session_data(self, session_id, **kwargs):
            if self.accesses["get_session_data"] < 1:
                raise ValueError("bad news!")
            return super(InitiallyFailingApi, self).get_session_data(session_id, **kwargs)

    api = InitiallyFailingApi()
    cache = epc.EcephysProjectCache(manifest=man_path, fetch_api=api)

    sid = 12345
    session = cache.get_session_data(sid)
    assert session.ecephys_session_id == sid


def test_get_session_data_continual_failure(tmpdir_factory, mock_api):
    man_path = os.path.join(
        tmpdir_factory.mktemp("get_session_data"),
        "manifest.json"
    )

    class ContinuallyFailingApi(mock_api):
        def get_session_data(self, session_id, **kwargs):
            raise ValueError("bad news!")

    api = ContinuallyFailingApi()
    cache = epc.EcephysProjectCache(manifest=man_path, fetch_api=api)

    sid = 12345
    with pytest.raises(ValueError):
        _ = cache.get_session_data(sid)


def test_get_probe_lfp_data(tmpdir_factory, mock_api):
    man_path = os.path.join(
        tmpdir_factory.mktemp("get_lfp_data"),
        "manifest.json"
    )

    class InitiallyFailingApi(mock_api):
        def get_probe_lfp_data(self, probe_id, **kwargs):
            if self.accesses["get_probe_data"] < 1:
                raise ValueError("bad news!")
            return super(InitiallyFailingApi, self).get_probe_lfp_data(probe_id, **kwargs)

    api = InitiallyFailingApi()
    cache = epc.EcephysProjectCache(manifest=man_path, fetch_api=api)

    sid = 3
    pid = 11

    session = cache.get_session_data(sid)
    lfp_file = session.api._probe_nwbfile(pid)

    assert str(pid) == lfp_file.identifier


def test_get_probe_lfp_data_continually_failing(tmpdir_factory, mock_api):
    man_path = os.path.join(
        tmpdir_factory.mktemp("get_lfp_data"),
        "manifest.json"
    )

    class ContinuallyFailingApi(mock_api):
        def get_probe_lfp_data(self, probe_id, **kwargs):
            if True:
                raise ValueError("bad news!")

    api = ContinuallyFailingApi()
    cache = epc.EcephysProjectCache(manifest=man_path, fetch_api=api)

    sid = 3
    pid = 11

    with pytest.raises(ValueError):
        session = cache.get_session_data(sid)
        _ = session.api._probe_nwbfile(pid)


def test_from_lims_default(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_from_lims_default"))

    cache = epc.EcephysProjectCache.from_lims(
        manifest=os.path.join(tmpdir, "manifest.json")
    )
    assert isinstance(cache.fetch_api.app_engine, HttpEngine)
    assert cache.stream_writer is write_from_stream
    assert cache.fetch_api.app_engine.scheme == "http"
    assert cache.fetch_api.app_engine.host == "lims2"


def test_from_warehouse_default(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_from_warehouse_default"))

    cache = epc.EcephysProjectCache.from_warehouse(
        manifest=os.path.join(tmpdir, "manifest.json")
    )
    assert isinstance(cache.fetch_api.rma_engine, HttpEngine)
    assert cache.stream_writer is write_from_stream
    assert cache.fetch_api.rma_engine.scheme == "http"
    assert cache.fetch_api.rma_engine.host == "api.brain-map.org"


def test_init_default(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_init_default"))
    cache = epc.EcephysProjectCache(
        manifest=os.path.join(tmpdir, "manifest.json")
    )
    assert isinstance(cache.fetch_api.rma_engine, HttpEngine)
    assert cache.stream_writer is cache.fetch_api.rma_engine.write_bytes
    assert cache.fetch_api.rma_engine.scheme == "http"
    assert cache.fetch_api.rma_engine.host == "api.brain-map.org"


@pytest.mark.parametrize(
    ("cache_constructor, asynchronous, engine_attr, expected_engine,"
     "expected_scheme, expected_host, expected_stream_writer"), [
        (
            epc.EcephysProjectCache.from_lims, True,
            "app_engine", AsyncHttpEngine, "http", "lims2",
            write_bytes_from_coroutine
        ),
        (
            epc.EcephysProjectCache.from_warehouse, True,
            "rma_engine", AsyncHttpEngine, "http", "api.brain-map.org",
            write_bytes_from_coroutine
        ),
        (
            epc.EcephysProjectCache.from_warehouse, False,
            "rma_engine", HttpEngine, "http", "api.brain-map.org",
            write_from_stream
        ),
        (
            epc.EcephysProjectCache.from_lims, False,
            "app_engine", HttpEngine, "http", "lims2",
            write_from_stream
        ),
    ])
def test_stream_asynchronous_arg(
        cache_constructor, asynchronous, engine_attr, expected_engine,
        expected_scheme, expected_host, expected_stream_writer,
        tmpdir_factory):
    """ Ensure the proper stream engine is chosen from the `asynchronous`
    argument in the EcephysProjectCache constructors (using other default
    values)."""
    tmpdir = str(tmpdir_factory.mktemp("test_stream_async_args"))
    cache = cache_constructor(
        asynchronous=asynchronous,
        manifest=os.path.join(tmpdir, "manifest.json")
    )
    engine = getattr(cache.fetch_api, engine_attr)
    assert isinstance(engine, expected_engine)
    assert cache.stream_writer is expected_stream_writer
    assert engine.scheme == expected_scheme
    assert engine.host == expected_host


def test_stream_writer_method_default_correct(tmpdir_factory):
    """Checks that the stream_writer contained in the rma engine is used
    when one is not supplied to the __init__ method.
    """
    tmpdir = str(tmpdir_factory.mktemp("test_from_warehouse_default"))
    manifest = os.path.join(tmpdir, "manifest.json")
    cache = epc.EcephysProjectCache(stream_writer=None, manifest=manifest)
    assert cache.stream_writer == cache.fetch_api.rma_engine.write_bytes
