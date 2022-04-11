import pytest
import numpy as np

from functools import partial

from allensdk.brain_observatory import sync_utilities as su
from allensdk.brain_observatory.sync_dataset import Dataset


class MockDataset(Dataset):
    def __init__(self, path: str,
                 eye_tracking_timings, behavior_tracking_timings):
        # Note: eye_tracking_timings and behavior_tracking_timings are test
        # inputs that can be parametrized and do not exist in the real
        # `Dataset` class.
        self.eye_tracking_timings = eye_tracking_timings
        self.behavior_tracking_timings = behavior_tracking_timings

    def get_edges(self, kind, keys, units='seconds'):
        if keys == self.EYE_TRACKING_KEYS:
            return self.eye_tracking_timings
        elif keys == self.BEHAVIOR_TRACKING_KEYS:
            return self.behavior_tracking_timings


@pytest.fixture
def mock_dataset_fixture(request):
    test_params = {
        "eye_tracking_timings": [],
        "behavior_tracking_timings": []
    }
    test_params.update(request.param)
    return partial(MockDataset, **test_params)


@pytest.mark.parametrize('vs_times, expected', [
    [[0.016, 0.033, 0.051, 0.067, 3.0], [0.016, 0.033, 0.051, 0.067]]
])
def test_trim_discontiguous_vsyncs(vs_times, expected):
    obtained = su.trim_discontiguous_times(vs_times)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    "mock_dataset_fixture,sync_line_label_keys,expected",
    [
        ({"eye_tracking_timings": [0.020, 0.030, 0.040, 0.050, 3.0]},
         Dataset.EYE_TRACKING_KEYS, [0.020, 0.030, 0.040, 0.050]),

        ({"behavior_tracking_timings": [0.080, 0.090, 0.100, 0.110, 8.0]},
         Dataset.BEHAVIOR_TRACKING_KEYS, [0.08, 0.090, 0.100, 0.110])
    ],
    indirect=["mock_dataset_fixture"])
def test_get_synchronized_frame_times(monkeypatch, mock_dataset_fixture,
                                      sync_line_label_keys, expected):
    monkeypatch.setattr(su, "Dataset", mock_dataset_fixture)

    obtained = su.get_synchronized_frame_times(
        "dummy_path",
        sync_line_label_keys
    )
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    "mock_dataset_fixture,sync_line_label_keys,expected",
    [
        ({"eye_tracking_timings": [0.020, 0.030, 0.040, 0.050, 3.0]},
         Dataset.EYE_TRACKING_KEYS, [0.020, 0.030, 0.040, 0.050, 3.0]),

        ({"behavior_tracking_timings": [0.080, 0.090, 0.100, 0.110, 8.0]},
         Dataset.BEHAVIOR_TRACKING_KEYS, [0.08, 0.090, 0.100, 0.110, 8.0])
    ], indirect=["mock_dataset_fixture"])
def test_get_synchronized_frame_times_no_trim(
    monkeypatch, mock_dataset_fixture, sync_line_label_keys, expected
):
    monkeypatch.setattr(su, "Dataset", mock_dataset_fixture)

    obtained = su.get_synchronized_frame_times(
        "dummy_path",
        sync_line_label_keys,
        trim_after_spike=False
    )
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    "mock_dataset_fixture,sync_line_label_keys,expected",
    [
        ({"eye_tracking_timings":
            [0.020, 0.030, 3.0, 0.040, 0.050, 0.040]},
         Dataset.EYE_TRACKING_KEYS, [0.020, 0.030]),

        ({"behavior_tracking_timings":
            [0.080, 8.0, 0.090, 0.100, 0.110, 0.150, 0.085, 0.110, 0.13]},
         Dataset.BEHAVIOR_TRACKING_KEYS, [0.08])
    ], indirect=["mock_dataset_fixture"])
def test_get_synchronized_frame_times_trim_with_spike(
    monkeypatch, mock_dataset_fixture, sync_line_label_keys, expected
):
    monkeypatch.setattr(su, "Dataset", mock_dataset_fixture)

    obtained = su.get_synchronized_frame_times(
        "dummy_path", sync_line_label_keys
    )
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    "mock_dataset_fixture,sync_line_label_keys,expected",
    [
        ({"eye_tracking_timings": [3.0, 0.030, 0.040, 0.050]},
         Dataset.EYE_TRACKING_KEYS, []),

        ({"behavior_tracking_timings": [8.0, 0.080, 0.090, 0.100, 0.110]},
         Dataset.BEHAVIOR_TRACKING_KEYS, [])
    ], indirect=["mock_dataset_fixture"])
def test_get_synchronized_frame_times_trim_all(
    monkeypatch,
    mock_dataset_fixture,
    sync_line_label_keys,
    expected
):

    monkeypatch.setattr(su, "Dataset", mock_dataset_fixture)

    obtained = su.get_synchronized_frame_times(
        "dummy_path",
        sync_line_label_keys
    )
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    "mock_dataset_fixture,sync_line_label_keys,expected",
    [
        ({"eye_tracking_timings": [0.020, 0.030, 3.0, 0.050, 0.040]},
         Dataset.EYE_TRACKING_KEYS, [0.020, 0.030, 3.0, 0.050, 0.040]),

        ({"behavior_tracking_timings": [0.080, 8.0, 0.090, 0.100, 0.110]},
         Dataset.BEHAVIOR_TRACKING_KEYS, [0.080, 8.0, 0.090, 0.100, 0.110])
    ],
    indirect=["mock_dataset_fixture"])
def test_get_synchronized_frame_times_no_trim_with_spike(
    monkeypatch,
    mock_dataset_fixture,
    sync_line_label_keys,
    expected
):
    monkeypatch.setattr(su, "Dataset", mock_dataset_fixture)

    obtained = su.get_synchronized_frame_times(
        "dummy_path",
        sync_line_label_keys,
        trim_after_spike=False
    )
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize(
    "mock_dataset_fixture,sync_line_label_keys,expected",
    [
        ({"eye_tracking_timings": [0.020, 0.030, 0.040, 0.050, 3.0]},
         Dataset.EYE_TRACKING_KEYS, [0.020, 0.030, 0.050]),

        ({"behavior_tracking_timings": [0.080, 0.090, 0.100, 0.110, 8.0]},
         Dataset.BEHAVIOR_TRACKING_KEYS, [0.080, 0.090, 0.110])
    ], indirect=["mock_dataset_fixture"])
def test_get_synchronized_frame_times_drop_frame(
    monkeypatch, mock_dataset_fixture, sync_line_label_keys, expected
):
    monkeypatch.setattr(su, "Dataset", mock_dataset_fixture)

    obtained = su.get_synchronized_frame_times(
        "dummy_path",
        sync_line_label_keys,
        drop_frames=[2]
    )
    assert np.allclose(obtained, expected)
