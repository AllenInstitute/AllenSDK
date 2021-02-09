from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import h5py
import os
from contextlib import contextmanager

from allensdk.internal.api import OneResultExpectedError
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysLimsApi, BehaviorOphysLimsExtractor)
from allensdk.brain_observatory.behavior.mtrain import ExtendedTrialSchema
from marshmallow.schema import ValidationError


@contextmanager
def does_not_raise(enter_result=None):
    """
    Context to help parametrize tests that may raise errors.
    If we start supporting only python 3.7+, switch to
    contextlib.nullcontext
    """
    yield enter_result


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("ophys_experiment_id", [
    pytest.param(511458874),
])
def test_get_cell_roi_table(ophys_experiment_id):
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    assert len(api.get_cell_specimen_table()) == 128


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("ophys_experiment_id, compare_val", [
    pytest.param(789359614,
                 ("/allen/programs/braintv/production/visualbehavior/prod0"
                  "/specimen_756577249/behavior_session_789295700/"
                  "789220000.pkl")),
    pytest.param(0, None)
])
def test_get_behavior_stimulus_file(ophys_experiment_id, compare_val):

    if compare_val is None:
        expected_fail = False
        try:
            api = BehaviorOphysLimsApi(ophys_experiment_id)
            api.extractor.get_behavior_stimulus_file()
        except OneResultExpectedError:
            expected_fail = True
        assert expected_fail is True
    else:
        api = BehaviorOphysLimsApi(ophys_experiment_id)
        assert api.extractor.get_behavior_stimulus_file() == compare_val


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("ophys_experiment_id", [789359614])
def test_get_extended_trials(ophys_experiment_id):

    api = BehaviorOphysLimsApi(ophys_experiment_id)
    df = api.get_extended_trials()
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df.to_dict("records")
    data_list_cs_sc = ets.dump(data_list_cs)
    ets.load(data_list_cs_sc)

    df_fail = df.drop(["behavior_session_uuid"], axis=1)
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df_fail.to_dict("records")
    data_list_cs_sc = ets.dump(data_list_cs)
    try:
        ets.load(data_list_cs_sc)
        raise RuntimeError("This should have failed with "
                           "marshmallow.schema.ValidationError")
    except ValidationError:
        pass


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("ophys_experiment_id", [860030092])
def test_get_nwb_filepath(ophys_experiment_id):

    api = BehaviorOphysLimsApi(ophys_experiment_id)
    assert api.extractor.get_nwb_filepath() == (
        "/allen/programs/braintv/production/visualbehavior/prod0/"
        "specimen_823826986/ophys_session_859701393/"
        "ophys_experiment_860030092/behavior_ophys_session_860030092.nwb")


@pytest.mark.parametrize(
    "timestamps,plane_group,group_count,expected",
    [
        (np.ones(10), 1, 0, np.ones(10)),
        (np.ones(10), 1, 0, np.ones(10)),
        # middle
        (np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), 1, 3, np.ones(4)),
        # first
        (np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]), 0, 4, np.ones(3)),
        # last
        (np.array([0, 1, 0, 1, 0, 1, 0, 1]), 1, 2, np.ones(4)),
        # only one group
        (np.ones(10), 0, 1, np.ones(10))
    ]
)
def test_process_ophys_plane_timestamps(
        timestamps, plane_group, group_count, expected):
    actual = BehaviorOphysLimsApi._process_ophys_plane_timestamps(
        timestamps, plane_group, group_count)
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "plane_group, ophys_timestamps, dff_traces, expected, context",
    [
        (None, np.arange(10), np.arange(5).reshape(1, 5),
         np.arange(5), does_not_raise()),
        (None, np.arange(10), np.arange(20).reshape(1, 20),
         None, pytest.raises(RuntimeError)),
        (0, np.arange(10), np.arange(5).reshape(1, 5),
         np.arange(0, 10, 2), does_not_raise()),
        (0, np.arange(20), np.arange(5).reshape(1, 5),
         None, pytest.raises(RuntimeError))
    ],
    ids=["scientifica-truncate", "scientifica-raise", "mesoscope-good",
         "mesoscope-raise"]
)
def test_get_ophys_timestamps(monkeypatch, plane_group, ophys_timestamps,
                              dff_traces, expected, context):
    """Test the acquisition frame truncation only happens for
    non-mesoscope data (and raises error for scientifica data with
    longer trace frames than acquisition frames (ophys_timestamps))."""

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor,
                    "get_behavior_session_id", lambda x: 123)
        ctx.setattr(BehaviorOphysLimsExtractor, "_get_ids", lambda x: {})
        patched_extractor = BehaviorOphysLimsExtractor(123)

        api = BehaviorOphysLimsApi(extractor=patched_extractor)

    # Mocking any db calls
    monkeypatch.setattr(api, "get_sync_data",
                        lambda: {"ophys_frames": ophys_timestamps})
    monkeypatch.setattr(api, "get_raw_dff_data", lambda: dff_traces)
    monkeypatch.setattr(api.extractor, "get_imaging_plane_group",
                        lambda: plane_group)
    monkeypatch.setattr(api.extractor, "get_plane_group_count", lambda: 2)
    with context:
        actual = api.get_ophys_timestamps()
        if expected is not None:
            np.testing.assert_array_equal(expected, actual)


def test_dff_trace_order(monkeypatch, tmpdir):
    """
    Test that BehaviorOphysLimsApi.get_raw_dff_data can reorder
    ROIs to align with what is in the cell_specimen_table
    """

    out_fname = os.path.join(tmpdir, "dummy_dff_data.h5")
    rng = np.random.RandomState(1234)
    n_t = 100
    data = rng.random_sample((5, n_t))
    roi_names = np.array([5, 3, 4, 2, 1])
    with h5py.File(out_fname, "w") as out_file:
        out_file.create_dataset("data", data=data)
        out_file.create_dataset("roi_names", data=roi_names.astype(bytes))

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id
        self.get_behavior_session_id = 2

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor, "get_dff_file",
                    lambda *args: out_fname)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        ctx.setattr(BehaviorOphysLimsApi, "get_cell_roi_ids",
                    lambda *args: np.array([1, 2, 3, 4, 5]).astype(bytes))
        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        dff_traces = api.get_raw_dff_data()

    # compare the returned traces with the input data
    # mapping to the order of the monkeypatched cell_roi_id list
    np.testing.assert_array_almost_equal(
        dff_traces[0, :], data[4, :], decimal=10)
    np.testing.assert_array_almost_equal(
        dff_traces[1, :], data[3, :], decimal=10)
    np.testing.assert_array_almost_equal(
        dff_traces[2, :], data[1, :], decimal=10)
    np.testing.assert_array_almost_equal(
        dff_traces[3, :], data[2, :], decimal=10)
    np.testing.assert_array_almost_equal(
        dff_traces[4, :], data[0, :], decimal=10)


def test_dff_trace_exceptions(monkeypatch, tmpdir):
    """
    Test that BehaviorOphysLimsApi.get_raw_dff_data() raises exceptions when
    dff trace file and cell_specimen_table contain different ROI IDs
    """

    # check that an exception is raised if dff_traces has an ROI ID
    # that cell_specimen_table does not
    out_fname = os.path.join(tmpdir, "dummy_dff_data_for_exceptions.h5")
    rng = np.random.RandomState(1234)
    n_t = 100
    data = rng.random_sample((5, n_t))
    roi_names = np.array([5, 3, 4, 2, 1])
    with h5py.File(out_fname, "w") as out_file:
        out_file.create_dataset("data", data=data)
        out_file.create_dataset("roi_names", data=roi_names.astype(bytes))

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id
        self.get_behavior_session_id = 2

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor, "get_dff_file",
                    lambda *args: out_fname)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        ctx.setattr(BehaviorOphysLimsApi, "get_cell_roi_ids",
                    lambda *args: np.array([1, 3, 4, 5]).astype(bytes))
        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        with pytest.raises(RuntimeError):
            _ = api.get_raw_dff_data()

    # check that an exception is raised if the cell_specimen_table
    # has an ROI ID that dff_traces does not
    out_fname = os.path.join(tmpdir, "dummy_dff_data_for_exceptions2.h5")
    rng = np.random.RandomState(1234)
    n_t = 100
    data = rng.random_sample((5, n_t))
    roi_names = np.array([5, 3, 4, 2, 1])
    with h5py.File(out_fname, "w") as out_file:
        out_file.create_dataset("data", data=data)
        out_file.create_dataset("roi_names", data=roi_names.astype(bytes))

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id
        self.get_behavior_session_id = 2

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor, "get_dff_file",
                    lambda *args: out_fname)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        ctx.setattr(BehaviorOphysLimsApi, "get_cell_roi_ids",
                    lambda *args: np.array([1, 2, 3, 4, 5, 6]).astype(bytes))
        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        with pytest.raises(RuntimeError):
            _ = api.get_raw_dff_data()


def test_corrected_fluorescence_trace_order(monkeypatch, tmpdir):
    """
    Test that BehaviorOphysLimsApi.get_corrected_fluorescence_traces
    can reorder ROIs to align with what is in the cell_specimen_table
    """

    out_fname = os.path.join(tmpdir, "dummy_ftrace_data.h5")
    rng = np.random.RandomState(1234)
    n_t = 100
    data = rng.random_sample((5, n_t))
    roi_names = np.array([5, 3, 4, 2, 1])
    with h5py.File(out_fname, "w") as out_file:
        out_file.create_dataset("data", data=data)
        out_file.create_dataset("roi_names", data=roi_names.astype(bytes))

    cell_data = {"junk": [6, 7, 8, 9, 10],
                 "cell_roi_id": [b"1", b"2", b"3", b"4", b"5"]}

    cell_table = pd.DataFrame(data=cell_data,
                              index=pd.Index([10, 20, 30, 40, 50],
                                             name="cell_specimen_id"))

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id
        self.get_behavior_session_id = 2

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor,
                    "get_demix_file", lambda *args: out_fname)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        ctx.setattr(BehaviorOphysLimsApi, "get_ophys_timestamps",
                    lambda *args: np.zeros(n_t))
        ctx.setattr(BehaviorOphysLimsApi, "get_cell_specimen_table",
                    lambda *args: cell_table)
        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        f_traces = api.get_corrected_fluorescence_traces()

    # check that the f_traces data frame was correctly joined
    # on roi_id
    colname = "corrected_fluorescence"
    roi_to_dex = {1: 4, 2: 3, 3: 1, 4: 2, 5: 0}
    for ii, roi_id in enumerate([1, 2, 3, 4, 5]):
        cell = (f_traces.loc[f_traces.cell_roi_id
                == bytes(f"{roi_id}", "utf-8")])
        assert cell.index.values[0] == 10*roi_id
        np.testing.assert_array_almost_equal(cell[colname].values[0],
                                             data[roi_to_dex[roi_id]],
                                             decimal=10)


def test_corrected_fluorescence_trace_exceptions(monkeypatch, tmpdir):
    """
    Test that BehaviorOphysLimsApi.get_corrected_fluorescence_traces
    raises exceptions when the trace file and cell_specimen_table have
    different ROI IDs

    Check case where cell_specimen_table has an ROI that
    the fluorescence traces do not
    """

    out_fname = os.path.join(tmpdir, "dummy_ftrace_data_exc.h5")
    rng = np.random.RandomState(1234)
    n_t = 100
    data = rng.random_sample((4, n_t))
    roi_names = np.array([5, 3, 4, 2])
    with h5py.File(out_fname, "w") as out_file:
        out_file.create_dataset("data", data=data)
        out_file.create_dataset("roi_names", data=roi_names.astype(bytes))

    cell_data = {"junk": [6, 7, 8, 9, 10],
                 "cell_roi_id": [b"1", b"2", b"3", b"4", b"5"]}

    cell_table = pd.DataFrame(data=cell_data,
                              index=pd.Index([10, 20, 30, 40, 50],
                                             name="cell_specimen_id"))

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id
        self.get_behavior_session_id = 2

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor,
                    "get_demix_file", lambda *args: out_fname)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        ctx.setattr(BehaviorOphysLimsApi, "get_ophys_timestamps",
                    lambda *args: np.zeros(n_t))
        ctx.setattr(BehaviorOphysLimsApi, "get_cell_specimen_table",
                    lambda *args: cell_table)
        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        with pytest.raises(RuntimeError):
            _ = api.get_corrected_fluorescence_traces()


def test_corrected_fluorescence_trace_exceptions2(monkeypatch, tmpdir):
    """
    Test that BehaviorOphysLimsApi.get_corrected_fluorescence_traces
    raises exceptions when the trace file and cell_specimen_table have
    different ROI IDs

    Check case where fluorescence traces have an ROI that
    the cell_specimen_table does not
    """

    out_fname = os.path.join(tmpdir, "dummy_ftrace_data_exc2.h5")
    rng = np.random.RandomState(1234)
    n_t = 100
    data = rng.random_sample((5, n_t))
    roi_names = np.array([1, 5, 3, 4, 2])
    with h5py.File(out_fname, "w") as out_file:
        out_file.create_dataset("data", data=data)
        out_file.create_dataset("roi_names", data=roi_names.astype(bytes))

    cell_data = {"junk": [6, 7, 8, 9, 10, 11],
                 "cell_roi_id": [b"1", b"2", b"3", b"4", b"5", b"6"]}

    cell_table = pd.DataFrame(data=cell_data,
                              index=pd.Index([10, 20, 30, 40, 50, 60],
                                             name="cell_specimen_id"))

    def dummy_init(self, ophys_experiment_id, **kwargs):
        self.ophys_experiment_id = ophys_experiment_id
        self.get_behavior_session_id = 2

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, "__init__", dummy_init)
        ctx.setattr(BehaviorOphysLimsExtractor,
                    "get_demix_file", lambda *args: out_fname)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        ctx.setattr(BehaviorOphysLimsApi, "get_ophys_timestamps",
                    lambda *args: np.zeros(n_t))
        ctx.setattr(BehaviorOphysLimsApi, "get_cell_specimen_table",
                    lambda *args: cell_table)
        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        with pytest.raises(RuntimeError):
            _ = api.get_corrected_fluorescence_traces()


def test_eye_tracking_rig_geometry_returns_single_rig(monkeypatch):
    """
    This test tests that when there are multiple rig geometries for an experiment,
    that only the most recent is returned
    """
    def dummy_init(self, ophys_experiment_id):
        self.ophys_experiment_id = ophys_experiment_id

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysLimsExtractor, '__init__', dummy_init)
        patched_extractor = BehaviorOphysLimsExtractor(123)

        api = BehaviorOphysLimsApi(extractor=patched_extractor)

        resources_dir = Path(os.path.dirname(__file__)) / 'resources'
        rig_geometry = (
            pd.read_pickle(resources_dir
                           / 'rig_geometry_multiple_rig_configs.pkl'))
        rig_geometry = api.extractor._process_eye_tracking_rig_geometry(
            rig_geometry=rig_geometry)

    expected = {
        'camera_position_mm': [102.8, 74.7, 31.6],
        'led_position': [246.0, 92.3, 52.6],
        'monitor_position_mm': [118.6, 86.2, 31.6],
        'camera_rotation_deg': [0.0, 0.0, 2.8],
        'monitor_rotation_deg': [0.0, 0.0, 0.0],
        'equipment': 'CAM2P.5'
    }

    assert rig_geometry == expected


@pytest.mark.requires_bamboo
def test_rig_geometry_newer_than_experiment():
    """
    This test ensures that if the experiment date_of_acquisition
    is before a rig activate_date that it is not returned as the rig
    used for the experiment
    """
    # This experiment has rig config more recent than the
    # experiment date_of_acquisition
    ophys_experiment_id = 521405260
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    rig_geometry = api.get_eye_tracking_rig_geometry()

    expected = {
        'camera_position_mm': [130.0, 0.0, 0.0],
        'led_position': [265.1, -39.3, 1.0],
        'monitor_position_mm': [170.0, 0.0, 0.0],
        'camera_rotation_deg': [0.0, 0.0, 13.1],
        'monitor_rotation_deg': [0.0, 0.0, 0.0],
        'equipment': 'CAM2P.1'
    }
    assert rig_geometry == expected
