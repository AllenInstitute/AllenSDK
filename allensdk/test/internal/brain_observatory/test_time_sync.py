import pytest
import numpy as np
import json
import os
import h5py
from pkg_resources import resource_filename
from mock import patch
from allensdk.internal.brain_observatory import time_sync as ts
from allensdk.internal.pipeline_modules import run_ophys_time_sync
from allensdk.brain_observatory.sync_dataset import Dataset


ASSUMED_DELAY = 0.0351


data_file = resource_filename(__name__, "time_sync_test_data.json")
test_data = json.load(open(data_file, "r"))

data_skip = False
if not os.path.exists(test_data["nikon"]["sync_file"]):
    data_skip = True

### Functions from lims2_modules ophys_time_sync.py for regression testing

MIN_BOUND = .03
MAX_BOUND = .04


### Mock keys
mock_keys = {
    "photodiode": "photodiode",
    "2p": "2p_vsync",
    "stimulus": "stim_vsync",
    "eye_camera": "cam2_exposure",
    "behavior_camera": "cam1_exposure",
    "acquiring": "2p_acquiring",
    "lick_sensor": "lick_1"
}

class MockSyncDataset(Dataset):
    """
    Mock the Dataset class so it doesn't load an h5 file upon
    initialization.
    """
    def __init__(self, data, line_labels=None):
        self.dfile = data
        self.line_labels=line_labels


def mock_get_real_photodiode_events(data, key):
    return data


def mock_get_events_by_line(line, units="seconds"):
    return line


def calculate_stimulus_alignment(stim_time, valid_twop_vsync_fall):
    stimulus_alignment = np.empty(len(stim_time))

    for index in range(len(stim_time)):
        crossings = np.nonzero(np.ediff1d(np.sign(valid_twop_vsync_fall - stim_time[index])) > 0)
        try:
            stimulus_alignment[index] = int(crossings[0][0])
        except:
            stimulus_alignment[index] = np.NaN

    return stimulus_alignment


def calculate_valid_twop_vsync_fall(sync_data, sample_frequency):
    twop_vsync_fall = sync_data.get_falling_edges('2p_vsync') / sample_frequency

    if len(twop_vsync_fall) == 0:
        raise ValueError('Error: twop_vsync_fall length is 0, possible invalid, missing, and/or bad data')

    ophys_start = twop_vsync_fall[0]

    valid_twop_vsync_fall = twop_vsync_fall[np.where(twop_vsync_fall > ophys_start)[0]]

    return valid_twop_vsync_fall


def calculate_stim_vsync_fall(sync_data, sample_frequency):
    stim_vsync_fall = sync_data.get_falling_edges('stim_vsync')[0:] / sample_frequency 

    return stim_vsync_fall


def find_start(twop_vsync_fall):
    start_index = 0

    in_start_frames = True
    found_start = False

    prev_value = None
    index = 0
    for value in twop_vsync_fall:
        if not found_start:
            if prev_value != None:
                diff = value - prev_value

                if diff < MIN_BOUND or diff > MAX_BOUND:
                    if in_start_frames:
                        in_start_frames = False

                elif not in_start_frames:

                    found_start = True
                    start_index = index			

            prev_value = value
        index+= 1

    return start_index


def sync_camera_stimulus(sync_data, sample_frequency, camera, ophys_experiment_id):
        twop_vsync_fall = sync_data.get_falling_edges('2p_vsync') / sample_frequency

        if len(twop_vsync_fall) == 0:
            raise ValueError('Error: twop_vsync_fall length is 0, possible invalid, missing, and/or bad data')

        try:
            twop_acquiring = sync_data.get_rising_edges('2p_acquiring')
            ophys_start = twop_acquiring / sample_frequency
        except:
            ophys_start = [find_start(twop_vsync_fall)]

        twop_vsync_fall = twop_vsync_fall[np.where(twop_vsync_fall > ophys_start)[0]]

        cam_fall = None

        if camera == 1:
            cam_fall = sync_data.get_falling_edges('cam1_exposure') / sample_frequency
        elif camera == 2:
            cam_fall = sync_data.get_falling_edges('cam2_exposure') / sample_frequency
        else:
            raise ValueError('Error: camera value ' + str(camera) + ' is invalid')

        frames = np.zeros((len(twop_vsync_fall), 1))


        for i in range(len(frames)):
                crossings = np.nonzero(np.ediff1d(np.sign(cam_fall - twop_vsync_fall[i])) > 0)

                try:
                        frames[i] = crossings[0][0]
                except:
                        frames[i] = np.NaN
                        
        return frames

### End of regression functions


@pytest.fixture
def nikon_input():
    input_data = test_data["nikon"].copy()
    input_data.pop("ophys_experiment_id")
    return input_data


@pytest.fixture
def scientifica_input():
    input_data = test_data["scientifica"].copy()
    input_data.pop("ophys_experiment_id")
    return input_data


@pytest.fixture
def input_json(tmpdir_factory):
    output_file = str(tmpdir_factory.mktemp("test").join("output.h5"))
    input_data = test_data["nikon"].copy()
    input_data['output_file'] = output_file
    json_file = str(tmpdir_factory.mktemp("test").join("input.json"))
    with open(json_file, "w") as f:
        json.dump(input_data, f)

    return json_file


def test_get_alignment_array():
    bigger = np.linspace(0, 5, 300)
    smaller = np.linspace(0.2, 3, 50)

    alignment = ts.get_alignment_array(bigger, smaller)
    assert np.all(~np.isnan(alignment))
    assert np.all(bigger[alignment.astype(int)] < smaller)
    
    alignment = ts.get_alignment_array(smaller, bigger)
    assert np.all(np.isnan(alignment[bigger <= 0.2]))
    assert np.all(np.isnan(alignment[bigger >= 50]))
    big_idx = np.where(~np.isnan(alignment))[0]
    small_idx = alignment[big_idx].astype(int)
    assert np.all(smaller[small_idx] < bigger[big_idx])


@pytest.mark.skipif(data_skip, reason="No sync or data")
def test_regression_valid_2p_timestamps(nikon_input, scientifica_input):
    sync_file = nikon_input.pop("sync_file")
    aligner = ts.OphysTimeAligner(sync_file, **nikon_input)
    freq = aligner.dataset.meta_data['ni_daq']['counter_output_freq']
    old_times = calculate_valid_twop_vsync_fall(aligner.dataset, freq)
    new_times = aligner.ophys_timestamps
    assert np.allclose(new_times[1:], old_times)

    # old scientifica used falling edges as timestamps incorrectly
    sync_file = scientifica_input.pop("sync_file")
    aligner = ts.OphysTimeAligner(sync_file, **scientifica_input)
    freq = aligner.dataset.meta_data['ni_daq']['counter_output_freq']
    old_times = calculate_valid_twop_vsync_fall(aligner.dataset, freq)
    new_times = aligner.ophys_timestamps
    assert len(new_times) - len(old_times) == 1
    assert np.all(new_times[1:] < old_times)
    assert np.all(new_times[2:] > old_times[:-1])


@pytest.mark.skipif(data_skip, reason="No sync or data")
def test_regression_stim_timestamps(nikon_input, scientifica_input):
    for input_data in [nikon_input, scientifica_input]:
        sync_file = input_data.pop("sync_file")
        aligner = ts.OphysTimeAligner(sync_file, **input_data)
        freq = aligner.dataset.meta_data['ni_daq']['counter_output_freq']
        old_times = calculate_stim_vsync_fall(aligner.dataset, freq)
        assert np.allclose(aligner.stim_timestamps, old_times)


@pytest.mark.skipif(data_skip, reason="No sync or data")
def test_regression_calculate_stimulus_alignment(nikon_input,
                                                 scientifica_input):
    for input_data in [nikon_input, scientifica_input]:
        sync_file = input_data.pop("sync_file")
        aligner = ts.OphysTimeAligner(sync_file, **input_data)
        old_align = calculate_stimulus_alignment(aligner.stim_timestamps,
                                                 aligner.ophys_timestamps)
        new_align = ts.get_alignment_array(aligner.ophys_timestamps,
                                           aligner.stim_timestamps)
        
        # Old alignment assigned simultaneous stim frames to the previous ophys
        # frame. Methods should only differ when ophys and stim are identical.
        mismatch = old_align != new_align
        mis_o = aligner.ophys_timestamps[new_align[mismatch].astype(int)]
        mis_s = aligner.stim_timestamps[mismatch]
        assert np.all(mis_o == mis_s)
        # Occurence of mismatch should be rare
        assert len(mis_o) < 0.005*len(aligner.ophys_timestamps)


@pytest.mark.skipif(data_skip, reason="No sync or data")
def test_regression_calculate_camera_alignment(nikon_input,
                                               scientifica_input):
    for input_data in [nikon_input, scientifica_input]:
        sync_file = input_data.pop("sync_file")
        aligner = ts.OphysTimeAligner(sync_file, **input_data)
        freq = aligner.dataset.meta_data['ni_daq']['counter_output_freq']
        old_eye_align = sync_camera_stimulus(aligner.dataset, freq, 2, 1)
        # old alignment throws out the first ophys timestamp
        new_eye_align = ts.get_alignment_array(aligner.eye_video_timestamps,
                                               aligner.ophys_timestamps[1:],
                                               int_method=np.ceil)
        mismatch = np.where(old_eye_align[:,0] != new_eye_align)
        mis_e = aligner.eye_video_timestamps[new_eye_align[mismatch].astype(int)]
        mis_o = aligner.ophys_timestamps[1:][mismatch]
        mis_o_plus = aligner.ophys_timestamps[1:][(mismatch[0]+1,)]
        # New method should only disagree when old method was wrong (old method
        # set an eye tracking frame to an earlier ophys frame). 
        assert np.all(mis_o < mis_e)
        assert np.all(mis_o_plus >= mis_e)
        # Occurence of mismatch should be rare
        assert len(mis_o) < 0.005*len(aligner.ophys_timestamps[1:])


@pytest.mark.parametrize("eye_data_length", (None, 5000, 6000))
def test_get_corrected_eye_times(eye_data_length):
    true_times = np.arange(6000)

    with patch.object(ts, "get_keys", return_value=mock_keys):
        with patch.object(ts.Dataset, "load"):
            aligner = ts.OphysTimeAligner("test")

    aligner.eye_data_length = eye_data_length
    with patch.object(ts.Dataset, "get_falling_edges",
                      return_value=true_times) as mock_falling:
        with patch("logging.info") as mock_log:
            times, delta = aligner.corrected_eye_video_timestamps

    if eye_data_length != 6000:
        mock_log.assert_called_once()
    else:
        assert mock_log.call_count == 0

    mock_falling.assert_called_once()
    assert np.all(times == true_times)

    if eye_data_length is None:
        assert delta == 0
    else:
        assert delta == (len(true_times) - eye_data_length)


@pytest.mark.parametrize("behavior_data_length", (None, 5000, 6000))
def test_get_corrected_behavior_times(behavior_data_length):
    true_times = np.arange(6000)

    with patch.object(ts, "get_keys", return_value=mock_keys):
        with patch.object(ts.Dataset, "load"):
            aligner = ts.OphysTimeAligner("test")

    aligner.behavior_data_length = behavior_data_length
    with patch.object(ts.Dataset, "get_falling_edges",
                      return_value=true_times) as mock_falling:
        with patch("logging.info") as mock_log:
            times, delta = aligner.corrected_behavior_video_timestamps

    if behavior_data_length != 6000:
        mock_log.assert_called_once()
    else:
        assert mock_log.call_count == 0

    mock_falling.assert_called_once()
    assert np.all(times == true_times)

    if behavior_data_length is None:
        assert delta == 0
    else:
        assert delta == (len(true_times) - behavior_data_length)


@pytest.mark.parametrize("stim_data_length,start_delay", [
    (None, False),
    (None, True),
    (5000, False),
    (5000, True),
    (6000, False),
    (6000, True)
    ])
def test_get_corrected_stim_times(stim_data_length, start_delay):
    true_falling = np.arange(0, 60, 0.01)
    true_rising = true_falling + 0.005
    if start_delay:
        true_falling[0] -= 3
        true_rising[0] -= 3

    with patch.object(ts, "get_keys", return_value=mock_keys):
        with patch.object(ts.Dataset, "load"):
            aligner = ts.OphysTimeAligner("test")

    aligner.stim_data_length = stim_data_length
    with patch.object(ts, "monitor_delay", return_value=ASSUMED_DELAY):
        with patch.object(ts.Dataset, "get_falling_edges",
                          return_value=true_falling) as mock_falling:
            with patch.object(ts.Dataset, "get_rising_edges",
                          return_value=true_rising) as mock_rising:
                with patch("logging.info") as mock_log:
                    times, delta, stim_delay = aligner.corrected_stim_timestamps

    if stim_data_length is None:
        mock_log.assert_called_once()
        assert mock_rising.call_count == 0
        assert delta == 0
    elif stim_data_length != len(true_falling) and start_delay:
        mock_rising.assert_called_once()
        assert mock_log.call_count == 2
        assert len(times) == len(true_falling) - 1
        assert delta == len(true_falling) - 1 - stim_data_length
        assert np.all(times == true_falling[1:] + ASSUMED_DELAY)
    elif stim_data_length != len(true_falling):
        mock_rising.assert_called_once()
        mock_log.assert_called_once()
        assert delta == len(true_falling) - stim_data_length
        assert np.all(times == true_falling + ASSUMED_DELAY)
    else:
        assert mock_rising.call_count == 0
        assert np.all(times == true_falling + ASSUMED_DELAY)
        assert mock_log.call_count == 0
        assert delta == 0


@pytest.mark.parametrize("ophys_data_length", (None, 5000, 6000, 7000))
def test_get_corrected_ophys_times_nikon(ophys_data_length):
    true_times = np.arange(6000)

    with patch.object(ts, "get_keys", return_value=mock_keys):
        with patch.object(ts.Dataset, "load"):
            aligner = ts.OphysTimeAligner("test", "NIKONA1RMP")

    aligner.ophys_data_length = ophys_data_length
    with patch.object(ts.Dataset, "get_falling_edges",
                      return_value=true_times) as mock_times:
        with patch.object(ts.Dataset, "get_rising_edges",
                          return_value=[0]) as mock_acquiring:
            with patch("logging.info") as mock_log:
                if ophys_data_length is not None and \
                   ophys_data_length > len(true_times):
                    with pytest.raises(ValueError):
                        times, delta = aligner.corrected_ophys_timestamps
                else:
                    times, delta = aligner.corrected_ophys_timestamps
                    if ophys_data_length is None:
                        assert np.all(times == true_times)
                        mock_log.assert_called_once()
                        assert delta == 0
                    elif ophys_data_length != len(true_times):
                        assert np.all(times == true_times[:-delta])
                        mock_log.assert_called_once()
                    else:
                        assert mock_log.call_count == 0
                        assert np.all(times == true_times)

    aligner.scanner = "bad"
    with pytest.raises(ValueError):
        aligner.corrected_ophys_timestamps


@pytest.mark.skipif(data_skip, reason="No sync or data")
def test_module(input_json):
    with patch("sys.argv", ["test_run", input_json]):
        with patch("logging.info") as mock_logging:
            run_ophys_time_sync.main()

    with open(input_json, "r") as f:
        input_data = json.load(f)

    output_file = input_data.pop("output_file")
    assert os.path.exists(output_file)

    input_data.pop("ophys_experiment_id")
    sync_file = input_data.pop("sync_file")
    aligner = ts.OphysTimeAligner(sync_file, **input_data)
    with h5py.File(output_file) as f:
        t, d = aligner.corrected_ophys_timestamps
        assert np.all(t == f['twop_vsync_fall'].value)
        assert np.all(d == f['ophys_delta'].value)
        st, sd, stim_delay = aligner.corrected_stim_timestamps
        align = ts.get_alignment_array(t, st)
        assert np.allclose(align, f['stimulus_alignment'].value,
                           equal_nan=True)
        assert np.all(sd == f['stim_delta'].value)
        et, ed = aligner.corrected_eye_video_timestamps
        align = ts.get_alignment_array(et, t, int_method=np.ceil)
        assert np.allclose(align, f['eye_tracking_alignment'].value,
                           equal_nan=True)
        assert np.all(ed == f['eye_delta'].value)
        bt, bd = aligner.corrected_behavior_video_timestamps
        align = ts.get_alignment_array(bt, t, int_method=np.ceil)
        assert np.allclose(align, f['body_camera_alignment'].value,
                           equal_nan=True)


@pytest.mark.parametrize(
    "sync_dset,stim_times,transition_interval,expected",
    [
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
         np.array([0.99, 1.99, 2.99, 3.99, 4.99]), 1, 0.01),
        (np.array([1.0, 2.0, 3.0, 4.0]),
         np.array([0.95, 2.0, 2.95, 4.0]), 1, 0.025),
        (np.array([1.0]), np.array([1.0]), 1, 0.0)
    ],
)
def test_monitor_delay(sync_dset, stim_times, transition_interval, expected,
                       monkeypatch):
    monkeypatch.setattr(ts, "get_real_photodiode_events",
                        mock_get_real_photodiode_events)
    pytest.approx(expected, ts.monitor_delay(sync_dset, stim_times, "key",
                                             transition_interval))


@pytest.mark.parametrize(
    "sync_dset,stim_times,transition_interval",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([0.9, 1.9, 2.9]), 1,),    # negative
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 2.1, 3.1, 4.1]), 1,),  # too big
    ],
)
def test_monitor_delay_raises_error(
        sync_dset, stim_times, transition_interval,
        monkeypatch):
    monkeypatch.setattr(ts, "get_real_photodiode_events",
                        mock_get_real_photodiode_events)
    with pytest.raises(ValueError):
        ts.monitor_delay(sync_dset, stim_times, "key", transition_interval)


@pytest.mark.parametrize(
    "arr,cond,n,expected",
    [
        (np.array([1, 1, 1, 2]), lambda x: x < 2, 3, 0),
        (np.array([2, 1, 1, 1]), lambda x: x < 2, 3, 1),
        (np.array([1, 2, 2, 1, 1]), lambda x: x >= 1, 2, 0),
        (np.array([]), lambda x: x < 1, 1, None),
        (np.array([1, 2, 3]), lambda x: x < 3, 4, None),
        (np.array([1, 2, 2, 3, 2]), lambda x: x == 2, 3, None),
    ]
)
def test_find_n(arr, cond, n, expected):
    assert expected == ts._find_n(arr, n, cond)


@pytest.mark.parametrize(
    "arr,cond,n,expected",
    [
        (np.array([1, 1, 1, 2]), lambda x: x < 2, 3, 2),
        (np.array([2, 1, 1, 1]), lambda x: x < 2, 3, 3),
        (np.array([1, 2, 2, 1, 1]), lambda x: x >= 1, 2, 4),
        (np.array([]), lambda x: x < 1, 1, None),
        (np.array([1, 2, 3]), lambda x: x < 3, 4, None),
        (np.array([1, 2, 2, 3, 2]), lambda x: x == 2, 3, None),
    ]
)
def test_find_last_n(arr, cond, n, expected):
    assert expected == ts._find_last_n(arr, n, cond)


@pytest.mark.parametrize(
    "sync_dset,expected",
    [
        ([0.25, 0.5, 0.75, 1., 2., 3., 5., 5.75], [1., 2., 3.]),
        ([1., 2., 3., 4.], [1., 2., 3., 4.]),
        ([0.25, 1., 2., 2.1, 2.2, 3., 4., 5.], [3., 4., 5.]),   # false alarm start
        ([0.25, 1., 2., 3., 4., 4.5, 5.1, 6.1], [1., 2., 3., 4.]),  # false alarm end
    ],
)
def test_get_photodiode_events(sync_dset, expected, monkeypatch):
    ds = MockSyncDataset(sync_dset)
    monkeypatch.setattr(ds, "get_events_by_line", mock_get_events_by_line)
    np.testing.assert_array_equal(
        expected, ts.get_photodiode_events(ds, sync_dset))


@pytest.mark.parametrize(
    "sync_dset,",
    [
        ([]),
        ([0.25, 0.25]),
        ([1., 2.]),
    ]
)
def test_photodiode_events_error_if_none_found(sync_dset, monkeypatch):
    ds = MockSyncDataset(sync_dset)
    monkeypatch.setattr(ds, "get_events_by_line", mock_get_events_by_line)
    with pytest.raises(ValueError):
        ts.get_photodiode_events(ds, sync_dset)


@pytest.mark.parametrize("deserialized_pkl,expected", [
    ({"vsynccount": 100}, 100),
    ({"items": {"behavior": {"intervalsms": [2, 2, 2, 2, 2]}}}, 6),
    ({"vsynccount": 20, "items": {"behavior": {"intervalsms": [3, 3]}}}, 20)
])
def test_get_stim_data_length(monkeypatch, deserialized_pkl, expected):
    def mock_read_pickle(*args, **kwargs):
        return deserialized_pkl

    monkeypatch.setattr(ts.pd, "read_pickle", mock_read_pickle)
    obtained = ts.get_stim_data_length("dummy_filepath")

    assert obtained == expected


@pytest.mark.parametrize("sync_dset, line_labels, expected_line_labels,"
                         "expected_log",
                         [(None, ['2p_vsync', 'stim_vsync', 'stim_photodiode',
                                  'acq_trigger', '', 'cam1_exposure',
                                  'cam2_exposure', 'lick_sensor'],
                          {
                            "photodiode": "stim_photodiode",
                            "2p": "2p_vsync",
                            "stimulus": "stim_vsync",
                            "eye_camera": "cam2_exposure",
                            "behavior_camera": "cam1_exposure",
                            "lick_sensor": "lick_sensor"},
                           [('root', 30, 'Could not find valid lines for the '
                                         'following data sources'),
                            ('root', 30, "acquiring (valid line label(s) = "
                                         "['2p_acquiring']")]),
                          (None, ['2p_vsync', 'stim_vsync', 'photodiode',
                                  'acq_trigger', 'behavior_monitoring',
                                  'eye_tracking', 'lick_1'],
                          {
                            "photodiode": "photodiode",
                            "2p": "2p_vsync",
                            "stimulus": "stim_vsync",
                            "eye_camera": "eye_tracking",
                            "behavior_camera": "behavior_monitoring",
                            "lick_sensor": "lick_1"},
                           [('root', 30, 'Could not find valid lines for the '
                                         'following data sources'),
                            ('root', 30, "acquiring (valid line label(s) = "
                                         "['2p_acquiring']")]),
                          (None, ['2p_vsync', 'stim_vsync', 'photodiode',
                                  'acq_trigger', '', 'behavior_monitoring',
                                  'lick_1'],
                          {
                            "photodiode": "photodiode",
                            "2p": "2p_vsync",
                            "stimulus": "stim_vsync",
                            "behavior_camera": "behavior_monitoring",
                            "lick_sensor": "lick_1"},
                           [('root', 30, 'Could not find valid lines for the '
                                         'following data sources'),
                            ('root', 30, "eye_camera (valid line label(s) = "
                                         "['cam2_exposure', 'eye_tracking', "
                                         "'eye_frame_received']"),
                            ('root', 30, "acquiring (valid line label(s) = "
                                         "['2p_acquiring']")]),
                          (None, [],
                          {},
                           [('root', 30, 'Could not find valid lines for the '
                                         'following data sources'),
                            ('root', 30, "photodiode (valid line label(s) = "
                                          "['stim_photodiode', 'photodiode']"),
                            ('root', 30, "2p (valid line label(s) = "
                                          "['2p_vsync']"),
                            ('root', 30, "stimulus (valid line label(s) = "
                                          "['stim_vsync', 'vsync_stim']"),
                            ('root', 30, "eye_camera (valid line label(s) = "
                                          "['cam2_exposure', 'eye_tracking', "
                                          "'eye_frame_received']"),
                            ('root', 30, "behavior_camera (valid line label(s) "
                                         "= ['cam1_exposure', "
                                         "'behavior_monitoring', "
                                         "'beh_frame_received']"),
                            ('root', 30, "acquiring (valid line label(s) = "
                                         "['2p_acquiring']"),
                            ('root', 30, "lick_sensor (valid line label(s) = "
                                         "['lick_1', 'lick_sensor']")]),
                          (None, ['', 'stim_vsync', 'photodiode', 'acq_trigger',
                                  'eye_tracking', 'lick_1', 'acq_trigger',
                                  'cam1_exposure'],
                          {
                            "photodiode": "photodiode",
                            "stimulus": "stim_vsync",
                            "eye_camera": "eye_tracking",
                            "behavior_camera": "cam1_exposure",
                            "lick_sensor": "lick_1"},
                           [('root', 30, 'Could not find valid lines for the '
                                         'following data sources'),
                            ('root', 30, "2p (valid line label(s) = "
                                         "['2p_vsync']"),
                            ('root', 30, "acquiring (valid line label(s) = "
                                         "['2p_acquiring']")]),
                          (None, ['barcode_ephys', 'vsync_stim',
                                  'stim_photodiode', 'stim_running',
                                  'beh_frame_received', 'eye_frame_received',
                                  'face_frame_received', 'stim_running_opto',
                                  'stim_trial_opto', 'face_came_frame_readout',
                                  'eye_cam_frame_readout',
                                  'beh_cam_frame_readout', 'face_cam_exposing',
                                  'eye_cam_exposing', 'beh_cam_exposing',
                                  'lick_sensor'],
                          {
                            "photodiode": "stim_photodiode",
                            "stimulus": "vsync_stim",
                            "eye_camera": "eye_frame_received",
                            "behavior_camera": "beh_frame_received",
                            "lick_sensor": "lick_sensor"},
                           [('root', 30, 'Could not find valid lines for the '
                                         'following data sources'),
                            ('root', 30, "2p (valid line label(s) = "
                                         "['2p_vsync']"),
                            ('root', 30, "acquiring (valid line label(s) = "
                                         "['2p_acquiring']")])
])
def test_get_keys(sync_dset, line_labels, expected_line_labels, expected_log,
                  caplog):
    """
    Test Cases:
        1) Test Case with V2 keys
        2) Test Case with V1 keys
        3) Test Case with eye camera key missing
        4) Test Case with all keys missing
        5) Test Case with 2p key missing
        6) Test Case with V3 keys

    """
    ds = MockSyncDataset(None, line_labels)
    keys = ts.get_keys(ds)
    assert keys == expected_line_labels
    assert caplog.record_tuples == expected_log


