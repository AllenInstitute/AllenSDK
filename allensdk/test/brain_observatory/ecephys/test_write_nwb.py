import os
from datetime import datetime, timezone
from pathlib import Path
import logging
import platform

import pytest
import pynwb
import pandas as pd
import numpy as np
import xarray as xr

from pynwb import NWBFile, NWBHDF5IO

from allensdk.brain_observatory.ecephys.current_source_density.__main__ import write_csd_to_h5
import allensdk.brain_observatory.ecephys.write_nwb.__main__ as write_nwb
from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysNwbSessionApi
from allensdk.test.brain_observatory.behavior.test_eye_tracking_processing import create_preload_eye_tracking_df


@pytest.fixture
def units_table():
    return pynwb.misc.Units.from_dataframe(pd.DataFrame({
        "peak_channel_id": [5, 10, 15],
        "local_index": [0, 1, 2],
        "quality": ["good", "good", "noise"],
        "firing_rate": [0.5, 1.2, -3.14],
        "snr": [1.0, 2.4, 5],
        "isi_violations": [34, 39, 22]
    }, index=pd.Index([11, 22, 33], name="id")), name="units")


@pytest.fixture
def spike_times():
    return {
        11: [1, 2, 3, 4, 5, 6],
        22: [],
        33: [13, 4, 12]
    }


@pytest.fixture
def running_speed():
    return pd.DataFrame({
        "start_time": [1., 2., 3., 4., 5.],
        "end_time": [2., 3., 4., 5., 6.],
        "velocity": [-1., -2., -1., 0., 1.],
        "net_rotation": [-np.pi, -2 * np.pi, -np.pi, 0, np.pi]
    })


@pytest.fixture
def raw_running_data():
    return pd.DataFrame({
        "frame_time": np.random.rand(4),
        "dx": np.random.rand(4),
        "vsig": np.random.rand(4),
        "vin": np.random.rand(4),
    })


@pytest.fixture
def stimulus_presentations_color():
    return pd.DataFrame({
        "alpha": [0.5, 0.4, 0.3, 0.2, 0.1],
        "start_time": [1., 2., 4., 5., 6.],
        "stop_time": [2., 4., 5., 6., 8.],
        "stimulus_name": ['gabors', 'gabors', 'random', 'movie', 'gabors'],
        "color": ["1.0", "", r"[1.0,-1.0, 10., -42.12, -.1]", "-1.0", ""]
    }, index=pd.Index(name="stimulus_presentations_id", data=[0, 1, 2, 3, 4]))


def test_roundtrip_basic_metadata(roundtripper):
    dt = datetime.now(timezone.utc)
    nwbfile = pynwb.NWBFile(
        session_description="EcephysSession",
        identifier="{}".format(12345),
        session_start_time=dt
    )

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    assert 12345 == api.get_ecephys_session_id()
    assert dt == api.get_session_start_time()


@pytest.mark.parametrize("metadata, expected_metadata", [
    ({
        "specimen_name": "mouse_1",
        "age_in_days": 100.0,
        "full_genotype": "wt",
        "strain": "c57",
        "sex": "F",
        "stimulus_name": "brain_observatory_2.0",
        "donor_id": 12345,
        "species": "Mus musculus"},
     {
        "specimen_name": "mouse_1",
        "age_in_days": 100.0,
        "age": "P100D",
        "full_genotype": "wt",
        "strain": "c57",
        "sex": "F",
        "stimulus_name": "brain_observatory_2.0",
        "subject_id": "12345",
        "species": "Mus musculus"})
])
def test_add_metadata(nwbfile, roundtripper, metadata, expected_metadata):
    nwbfile = write_nwb.add_metadata_to_nwbfile(nwbfile, metadata)

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained = api.get_metadata()

    assert set(expected_metadata.keys()) == set(obtained.keys())

    misses = {}
    for key, value in expected_metadata.items():
        if obtained[key] != value:
            misses[key] = {"expected": value, "obtained": obtained[key]}

    assert len(misses) == 0, f"the following metadata items were mismatched: {misses}"


@pytest.mark.parametrize("presentations", [
    (pd.DataFrame({
        'alpha': [0.5, 0.4, 0.3, 0.2, 0.1],
        'start_time': [1., 2., 4., 5., 6.],
        'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
        'stop_time': [2., 4., 5., 6., 8.]
    }, index=pd.Index(name='stimulus_presentations_id', data=[0, 1, 2, 3, 4]))),

    (pd.DataFrame({
        'gabor_specific_column': [1.0, 2.0, np.nan, np.nan, 3.0],
        'mixed_column': ["a", "", "b", "", "c"],
        'movie_specific_column': [np.nan, np.nan, np.nan, 1.0, np.nan],
        'start_time': [1., 2., 4., 5., 6.],
        'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
        'stop_time': [2., 4., 5., 6., 8.]
    }, index=pd.Index(name='stimulus_presentations_id', data=[0, 1, 2, 3, 4]))),
])
def test_add_stimulus_presentations(nwbfile, presentations, roundtripper):
    write_nwb.add_stimulus_timestamps(nwbfile, [0, 1])
    write_nwb.add_stimulus_presentations(nwbfile, presentations)

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained_stimulus_table = api.get_stimulus_presentations()

    pd.testing.assert_frame_equal(presentations, obtained_stimulus_table, check_dtype=False)


def test_add_stimulus_presentations_color(nwbfile, stimulus_presentations_color, roundtripper):
    write_nwb.add_stimulus_timestamps(nwbfile, [0, 1])
    write_nwb.add_stimulus_presentations(nwbfile, stimulus_presentations_color)

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained_stimulus_table = api.get_stimulus_presentations()

    expected_color = [1.0, "", "", -1.0, ""]
    obtained_color = obtained_stimulus_table["color"].values.tolist()

    mismatched = False
    for expected, obtained in zip(expected_color, obtained_color):
        if expected != obtained:
            mismatched = True

    assert not mismatched, f"expected: {expected_color}, obtained: {obtained_color}"


@pytest.mark.parametrize("opto_table, expected", [
    (pd.DataFrame({
        "start_time": [0., 1., 2., 3.],
        "stop_time": [0.5, 1.5, 2.5, 3.5],
        "level": [10., 9., 8., 7.],
        "condition": ["a", "a", "b", "c"]}),
     None),

    # Test for older version of optotable that used nwb reserved "name" col
    (pd.DataFrame({"start_time": [0., 1., 2., 3.],
                   "stop_time": [0.5, 1.5, 2.5, 3.5],
                   "level": [10., 9., 8., 7.],
                   "condition": ["a", "a", "b", "c"],
                   "name": ["w", "x", "y", "z"]}),
     pd.DataFrame({"start_time": [0., 1., 2., 3.],
                   "stop_time": [0.5, 1.5, 2.5, 3.5],
                   "level": [10., 9., 8., 7.],
                   "condition": ["a", "a", "b", "c"],
                   "stimulus_name": ["w", "x", "y", "z"],
                   "duration": [0.5, 0.5, 0.5, 0.5]})),

    (pd.DataFrame({"start_time": [0., 1., 2., 3.],
                   "stop_time": [0.5, 1.5, 2.5, 3.5],
                   "level": [10., 9., 8., 7.],
                   "condition": ["a", "a", "b", "c"],
                   "stimulus_name": ["w", "x", "y", "z"]}),
     None)
])
def test_add_optotagging_table_to_nwbfile(nwbfile, roundtripper, opto_table, expected):

    opto_table["duration"] = opto_table["stop_time"] - opto_table["start_time"]

    nwbfile = write_nwb.add_optotagging_table_to_nwbfile(nwbfile, opto_table)
    api = roundtripper(nwbfile, EcephysNwbSessionApi)

    obtained = api.get_optogenetic_stimulation()

    if expected is None:
        expected = opto_table

    pd.testing.assert_frame_equal(obtained, expected, check_like=True)


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("pid,name,srate,lfp_srate,has_lfp,expected", [
    [
        12,
        "a probe",
        30000.0,
        2500.0,
        True,
        pd.DataFrame({
            "description": ["a probe"],
            "sampling_rate": [30000.0],
            "lfp_sampling_rate": [2500.0],
            "has_lfp_data": [True],
            "location": ["See electrode locations"]
        }, index=pd.Index([12], name="id"))
    ]
])
def test_add_probe_to_nwbfile(nwbfile, roundtripper, roundtrip, pid, name, srate, lfp_srate, has_lfp, expected):

    nwbfile, _, _ = write_nwb.add_probe_to_nwbfile(nwbfile, pid,
                                                   name=name,
                                                   sampling_rate=srate,
                                                   lfp_sampling_rate=lfp_srate,
                                                   has_lfp_data=has_lfp)
    if roundtrip:
        obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(expected, obt.get_probes(), check_like=True)


@pytest.mark.parametrize("columns_to_add, expected_columns", [
    (None,

     {"probe_vertical_position", "probe_horizontal_position",
      "probe_id", "local_index", "valid_data", "x", "y", "z", "group",
      "group_name", "imp", "location", "filtering"}),

    ([("test_column_a", "description_a"),
      ("test_column_b", "description_b")],

     {"x", "y", "z", "group", "group_name", "imp", "location", "filtering",
      "test_column_a", "test_column_b"})
])
def test_add_ecephys_electrode_columns(nwbfile, columns_to_add,
                                       expected_columns):

    write_nwb.add_ecephys_electrode_columns(nwbfile, columns_to_add)

    assert set(nwbfile.electrodes.colnames) == expected_columns


@pytest.mark.parametrize(("channels, local_index_whitelist, "
                          "expected_electrode_table"), [
    ([{"id": 1,
       "probe_id": 1234,
       "valid_data": True,
       "local_index": 23,
       "probe_vertical_position": 10,
       "probe_horizontal_position": 10,
       "anterior_posterior_ccf_coordinate": 15.0,
       "dorsal_ventral_ccf_coordinate": 20.0,
       "left_right_ccf_coordinate": 25.0,
       "manual_structure_acronym": "CA1",
       "impedence": np.nan,
       "filtering": "AP band: 500 Hz high-pass; LFP band: 1000 Hz low-pass"},
      {"id": 2,
       "probe_id": 1234,
       "valid_data": True,
       "local_index": 15,
       "probe_vertical_position": 20,
       "probe_horizontal_position": 20,
       "anterior_posterior_ccf_coordinate": 25.0,
       "dorsal_ventral_ccf_coordinate": 30.0,
       "left_right_ccf_coordinate": 35.0,
       "manual_structure_acronym": "CA3",
       "impedence": 42.0,
       "filtering": "custom"}],

     [15, 23],

     pd.DataFrame({
         "id": [2, 1],
         "probe_id": [1234, 1234],
         "valid_data": [True, True],
         "local_index": [15, 23],
         "probe_vertical_position": [20, 10],
         "probe_horizontal_position": [20, 10],
         "x": [25.0, 15.0],
         "y": [30.0, 20.0],
         "z": [35.0, 25.0],
         "location": ["CA3", "CA1"],
         "imp": [42.0, np.nan],
         "filtering": ["custom", "AP band: 500 Hz high-pass; LFP band: 1000 Hz low-pass"]
     }).set_index("id"))

])
def test_add_ecephys_electrodes(nwbfile, channels, local_index_whitelist,
                                expected_electrode_table):

    mock_device = pynwb.device.Device(name="mock_device")
    mock_electrode_group = pynwb.ecephys.ElectrodeGroup(name="mock_group",
                                                        description="",
                                                        location="",
                                                        device=mock_device)

    write_nwb.add_ecephys_electrodes(nwbfile, channels, mock_electrode_group,
                                     local_index_whitelist)

    obt_electrode_table = nwbfile.electrodes.to_dataframe().drop(columns=["group", "group_name"])

    pd.testing.assert_frame_equal(obt_electrode_table,
                                  expected_electrode_table,
                                  check_like=True)


@pytest.mark.parametrize("dc,order,exp_idx,exp_data", [
    [{"a": [1, 2, 3], "b": [4, 5, 6]}, ["a", "b"], [3, 6], [1, 2, 3, 4, 5, 6]]
])
def test_dict_to_indexed_array(dc, order, exp_idx, exp_data):

    obt_idx, obt_data = write_nwb.dict_to_indexed_array(dc, order)
    assert np.allclose(exp_idx, obt_idx)
    assert np.allclose(exp_data, obt_data)


def test_add_ragged_data_to_dynamic_table(units_table, spike_times):

    write_nwb.add_ragged_data_to_dynamic_table(
        table=units_table,
        data=spike_times,
        column_name="spike_times"
    )

    assert np.allclose([1, 2, 3, 4, 5, 6], units_table["spike_times"][0])
    assert np.allclose([], units_table["spike_times"][1])
    assert np.allclose([13, 4, 12], units_table["spike_times"][2])


@pytest.mark.parametrize("roundtrip,include_rotation", [
    [True, True],
    [True, False]
])
def test_add_running_speed_to_nwbfile(nwbfile, running_speed, roundtripper, roundtrip, include_rotation):

    nwbfile = write_nwb.add_running_speed_to_nwbfile(nwbfile, running_speed)
    if roundtrip:
        api_obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        api_obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    obtained = api_obt.get_running_speed(include_rotation=include_rotation)

    expected = running_speed
    if not include_rotation:
        expected = expected.drop(columns="net_rotation")
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize("roundtrip", [[True]])
def test_add_raw_running_data_to_nwbfile(nwbfile, raw_running_data, roundtripper, roundtrip):

    nwbfile = write_nwb.add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)
    if roundtrip:
        api_obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        api_obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    obtained = api_obt.get_raw_running_data()

    expected = raw_running_data.rename(columns={"dx": "net_rotation", "vsig": "signal_voltage", "vin": "supply_voltage"})
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize("presentations, column_renames_map, columns_to_drop, expected", [
    (pd.DataFrame({'alpha': [0.5, 0.4, 0.3, 0.2, 0.1],
                   'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
                   'start_time': [1., 2., 4., 5., 6.],
                   'stop_time': [2., 4., 5., 6., 8.]}),
     {"alpha": "beta"},
     None,
     pd.DataFrame({'beta': [0.5, 0.4, 0.3, 0.2, 0.1],
                   'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
                   'start_time': [1., 2., 4., 5., 6.],
                   'stop_time': [2., 4., 5., 6., 8.]})),

    (pd.DataFrame({'alpha': [0.5, 0.4, 0.3, 0.2, 0.1],
                   'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
                   'start_time': [1., 2., 4., 5., 6.],
                   'stop_time': [2., 4., 5., 6., 8.]}),
     {"alpha": "beta"},
     ["Nonexistant_column_to_drop"],
     pd.DataFrame({'beta': [0.5, 0.4, 0.3, 0.2, 0.1],
                   'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
                   'start_time': [1., 2., 4., 5., 6.],
                   'stop_time': [2., 4., 5., 6., 8.]})),

    (pd.DataFrame({'alpha': [0.5, 0.4, 0.3, 0.2, 0.1],
                   'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
                   'Start': [1., 2., 4., 5., 6.],
                   'End': [2., 4., 5., 6., 8.]}),
     None,
     ["alpha"],
     pd.DataFrame({'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
                   'start_time': [1., 2., 4., 5., 6.],
                   'stop_time': [2., 4., 5., 6., 8.]})),
])
def test_read_stimulus_table(tmpdir_factory, presentations,
                             column_renames_map, columns_to_drop, expected):
    dirname = str(tmpdir_factory.mktemp("ecephys_nwb_test"))
    stim_table_path = os.path.join(dirname, "stim_table.csv")

    presentations.to_csv(stim_table_path, index=False)
    obt = write_nwb.read_stimulus_table(stim_table_path,
                                        column_renames_map=column_renames_map,
                                        columns_to_drop=columns_to_drop)

    pd.testing.assert_frame_equal(obt, expected)


# read_spike_times_to_dictionary(spike_times_path, spike_units_path, local_to_global_unit_map=None)
def test_read_spike_times_to_dictionary(tmpdir_factory):
    dirname = str(tmpdir_factory.mktemp("ecephys_nwb_spike_times"))
    spike_times_path = os.path.join(dirname, "spike_times.npy")
    spike_units_path = os.path.join(dirname, "spike_units.npy")

    spike_times = np.sort(np.random.rand(30))
    np.save(spike_times_path, spike_times, allow_pickle=False)

    spike_units = np.concatenate([np.arange(15), np.arange(15)])
    np.save(spike_units_path, spike_units, allow_pickle=False)

    local_to_global_unit_map = {ii: -ii for ii in spike_units}

    obtained = write_nwb.read_spike_times_to_dictionary(spike_times_path, spike_units_path, local_to_global_unit_map)
    for ii in range(15):
        assert np.allclose(obtained[-ii], sorted([spike_times[ii], spike_times[15 + ii]]))


def test_read_waveforms_to_dictionary(tmpdir_factory):
    dirname = str(tmpdir_factory.mktemp("ecephys_nwb_mean_waveforms"))
    waveforms_path = os.path.join(dirname, "mean_waveforms.npy")

    nunits = 10
    nchannels = 30
    nsamples = 20

    local_to_global_unit_map = {ii: -ii for ii in range(nunits)}

    mean_waveforms = np.random.rand(nunits, nsamples, nchannels)
    np.save(waveforms_path, mean_waveforms, allow_pickle=False)

    obtained = write_nwb.read_waveforms_to_dictionary(waveforms_path, local_to_global_unit_map)
    for ii in range(nunits):
        assert np.allclose(mean_waveforms[ii, :, :], obtained[-ii])


@pytest.fixture
def lfp_data():
    total_timestamps = 12
    subsample_channels = np.array([3, 2])

    return {
        "data": np.arange(total_timestamps * len(subsample_channels), dtype=np.int16).reshape((total_timestamps, len(subsample_channels))),
        "timestamps": np.linspace(0, 1, total_timestamps),
        "subsample_channels": subsample_channels
    }


@pytest.fixture
def probe_data():
    probe_data = {
        "id": 12345,
        "name": "probeA",
        "sampling_rate": 29.0,
        "lfp_sampling_rate": 10.0,
        "temporal_subsampling_factor": 2.0,
        "channels": [
            {
                "id": 0,
                "probe_id": 12,
                "local_index": 1,
                "probe_vertical_position": 21,
                "probe_horizontal_position": 33,
                "valid_data": True,
                "anterior_posterior_ccf_coordinate": 5.0,
                "dorsal_ventral_ccf_coordinate": 10.0,
                "left_right_ccf_coordinate": 15.0,
                "manual_structure_acronym": "CA1",
                "impedence": np.nan,
                "filtering": "Unknown"
            },
            {
                "id": 1,
                "probe_id": 12,
                "local_index": 2,
                "probe_vertical_position": 21,
                "probe_horizontal_position": 32,
                "valid_data": True,
                "anterior_posterior_ccf_coordinate": 10.0,
                "dorsal_ventral_ccf_coordinate": 15.0,
                "left_right_ccf_coordinate": 20.0,
                "manual_structure_acronym": "CA2",
                "impedence": np.nan,
                "filtering": "Unknown"
            },
            {
                "id": 2,
                "probe_id": 12,
                "local_index": 3,
                "probe_vertical_position": 21,
                "probe_horizontal_position": 31,
                "valid_data": True,
                "anterior_posterior_ccf_coordinate": 15.0,
                "dorsal_ventral_ccf_coordinate": 20.0,
                "left_right_ccf_coordinate": 25.0,
                "manual_structure_acronym": "CA3",
                "impedence": np.nan,
                "filtering": "Unknown"
            }
        ],
        "lfp": {
            "input_data_path": "",
            "input_timestamps_path": "",
            "input_channels_path": "",
            "output_path": ""
        },
        "csd_path": "",
        "amplitude_scale_factor": 1.0
    }
    return probe_data


@pytest.fixture
def csd_data():
    csd_data = {
        "csd": np.arange(20).reshape([2, 10]),
        "relative_window": np.linspace(-1, 1, 10),
        "channels": np.array([3, 2]),
        "csd_locations": np.array([[1, 2], [3, 3]]),
        "stimulus_name": "foo",
        "stimulus_index": None,
        "num_trials": 1000
    }
    return csd_data


def test_write_probe_lfp_file(tmpdir_factory, lfp_data, probe_data, csd_data):

    tmpdir = Path(tmpdir_factory.mktemp("probe_lfp_nwb"))
    input_data_path = tmpdir / Path("lfp_data.dat")
    input_timestamps_path = tmpdir / Path("lfp_timestamps.npy")
    input_channels_path = tmpdir / Path("lfp_channels.npy")
    input_csd_path = tmpdir / Path("csd.h5")
    output_path = str(tmpdir / Path("lfp.nwb"))  # pynwb.NWBHDF5IO chokes on Path

    test_lfp_paths = {
        "input_data_path": input_data_path,
        "input_timestamps_path": input_timestamps_path,
        "input_channels_path": input_channels_path,
        "output_path": output_path
    }

    test_session_metadata = {
        "specimen_name": "A",
        "age_in_days": 100.0,
        "full_genotype": "wt",
        "strain": "A strain",
        "sex": "M",
        "stimulus_name": "test_stim",
        "species": "Mus musculus",
        "donor_id": 42
    }

    probe_data.update({"lfp": test_lfp_paths})
    probe_data.update({"csd_path": input_csd_path})

    write_csd_to_h5(path=input_csd_path, **csd_data)

    np.save(input_timestamps_path, lfp_data["timestamps"], allow_pickle=False)
    np.save(input_channels_path, lfp_data["subsample_channels"], allow_pickle=False)
    with open(input_data_path, "wb") as input_data_file:
        input_data_file.write(lfp_data["data"].tobytes())

    write_nwb.write_probe_lfp_file(4242, test_session_metadata, datetime.now(), logging.INFO, probe_data)

    exp_electrodes = pd.DataFrame(probe_data["channels"]).set_index("id").loc[[2, 1], :]
    exp_electrodes.rename(columns={"anterior_posterior_ccf_coordinate": "x",
                                   "dorsal_ventral_ccf_coordinate": "y",
                                   "left_right_ccf_coordinate": "z",
                                   "manual_structure_acronym": "location"}, inplace=True)

    with pynwb.NWBHDF5IO(output_path, "r") as obt_io:
        obt_f = obt_io.read()

        obt_ser = obt_f.get_acquisition("probe_12345_lfp").electrical_series["probe_12345_lfp_data"]
        assert np.allclose(lfp_data["data"], obt_ser.data[:])
        assert np.allclose(lfp_data["timestamps"], obt_ser.timestamps[:])

        obt_electrodes = obt_f.electrodes.to_dataframe().loc[
            :, ["local_index", "probe_horizontal_position",
                "probe_id", "probe_vertical_position",
                "valid_data", "x", "y", "z", "location", "impedence",
                "filtering"]
        ]

        assert obt_f.session_id == "4242"
        assert obt_f.subject.subject_id == "42"

        # There is a difference in how int dtypes are being saved in Windows
        # that are causing tests to fail.
        # Perhaps related to: https://stackoverflow.com/a/36279549
        if platform.system() == "Windows":
            pd.testing.assert_frame_equal(obt_electrodes, exp_electrodes, check_like=True, check_dtype=False)
        else:
            pd.testing.assert_frame_equal(obt_electrodes, exp_electrodes, check_like=True)

        csd_series = obt_f.get_processing_module("current_source_density")["current_source_density"]

        assert np.allclose(csd_data["csd"], csd_series.time_series.data[:].T)
        assert np.allclose(csd_data["relative_window"], csd_series.time_series.timestamps[:])
        obt_channel_locations = np.stack((csd_series.virtual_electrode_x_positions,
                                          csd_series.virtual_electrode_y_positions),
                                         axis=1)
        assert np.allclose([[1, 2], [3, 3]], obt_channel_locations)  # csd interpolated channel locations


@pytest.mark.parametrize("roundtrip", [True, False])
def test_write_probe_lfp_file_roundtrip(tmpdir_factory, roundtrip, lfp_data, probe_data, csd_data):

    expected_csd = xr.DataArray(
        name="CSD",
        data=csd_data["csd"],
        dims=["virtual_channel_index", "time"],
        coords={
            "virtual_channel_index": np.arange(csd_data["csd"].shape[0]),
            "time": csd_data["relative_window"],
            "vertical_position": (("virtual_channel_index",), csd_data["csd_locations"][:, 1]),
            "horizontal_position": (("virtual_channel_index",), csd_data["csd_locations"][:, 0]),
        }
    )

    expected_lfp = xr.DataArray(
        name="LFP",
        data=lfp_data["data"],
        dims=["time", "channel"],
        coords=[lfp_data["timestamps"], [2, 1]]
    )

    tmpdir = Path(tmpdir_factory.mktemp("probe_lfp_nwb"))
    input_data_path = tmpdir / Path("lfp_data.dat")
    input_timestamps_path = tmpdir / Path("lfp_timestamps.npy")
    input_channels_path = tmpdir / Path("lfp_channels.npy")
    input_csd_path = tmpdir / Path("csd.h5")
    output_path = str(tmpdir / Path("lfp.nwb"))  # pynwb.NWBHDF5IO chokes on Path

    test_lfp_paths = {
        "input_data_path": input_data_path,
        "input_timestamps_path": input_timestamps_path,
        "input_channels_path": input_channels_path,
        "output_path": output_path
    }

    probe_data.update({"lfp": test_lfp_paths})
    probe_data.update({"csd_path": input_csd_path})

    write_csd_to_h5(path=input_csd_path, **csd_data)

    np.save(input_timestamps_path, lfp_data["timestamps"], allow_pickle=False)
    np.save(input_channels_path, lfp_data["subsample_channels"], allow_pickle=False)
    with open(input_data_path, "wb") as input_data_file:
        input_data_file.write(lfp_data["data"].tobytes())

    write_nwb.write_probe_lfp_file(4242, None, datetime.now(), logging.INFO, probe_data)

    obt = EcephysNwbSessionApi(path=None, probe_lfp_paths={12345: NWBHDF5IO(output_path, "r").read})

    obtained_lfp = obt.get_lfp(12345)
    obtained_csd = obt.get_current_source_density(12345)

    xr.testing.assert_equal(obtained_lfp, expected_lfp)
    xr.testing.assert_equal(obtained_csd, expected_csd)


@pytest.fixture
def invalid_epochs():

    epochs = [
        {
            "type": "EcephysSession",
            "id": 739448407,
            "label": "stimulus",
            "start_time": 1998.0,
            "end_time": 2005.0,
        },
        {
            "type": "EcephysSession",
            "id": 739448407,
            "label": "stimulus",
            "start_time": 2114.0,
            "end_time": 2121.0,
        },
        {
            "type": "EcephysProbe",
            "id": 123448407,
            "label": "ProbeB",
            "start_time": 114.0,
            "end_time": 211.0,
        },
    ]

    return epochs


def test_add_invalid_times(invalid_epochs, tmpdir_factory):

    nwbfile_name = str(tmpdir_factory.mktemp("test").join("test_invalid_times.nwb"))

    nwbfile = NWBFile(
        session_description="EcephysSession",
        identifier="{}".format(739448407),
        session_start_time=datetime.now()
    )

    nwbfile = write_nwb.add_invalid_times(nwbfile, invalid_epochs)

    with NWBHDF5IO(nwbfile_name, mode="w") as io:
        io.write(nwbfile)
    nwbfile_in = NWBHDF5IO(nwbfile_name, mode="r").read()

    df = nwbfile.invalid_times.to_dataframe()
    df_in = nwbfile_in.invalid_times.to_dataframe()

    pd.testing.assert_frame_equal(df, df_in, check_like=True, check_dtype=False)


def test_roundtrip_add_invalid_times(nwbfile, invalid_epochs, roundtripper):

    expected = write_nwb.setup_table_for_invalid_times(invalid_epochs)

    nwbfile = write_nwb.add_invalid_times(nwbfile, invalid_epochs)
    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained = api.get_invalid_times()

    pd.testing.assert_frame_equal(expected, obtained, check_dtype=False)


def test_no_invalid_times_table():

    epochs = []
    assert write_nwb.setup_table_for_invalid_times(epochs).empty is True


def test_setup_table_for_invalid_times():

    epoch = {
        "type": "EcephysSession",
        "id": 739448407,
        "label": "stimulus",
        "start_time": 1998.0,
        "end_time": 2005.0,
    }

    s = write_nwb.setup_table_for_invalid_times([epoch]).loc[0]

    assert s["start_time"] == epoch["start_time"]
    assert s["stop_time"] == epoch["end_time"]
    assert s["tags"] == [epoch["type"], str(epoch["id"]), epoch["label"]]


@pytest.fixture
def spike_amplitudes():
    return np.arange(5)


@pytest.fixture
def templates():
    return np.array([
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [10, 21, 32]
        ],
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [15, 9, 4]
        ]
    ])


@pytest.fixture
def spike_templates():
    return np.array([0, 1, 0, 1, 0])


@pytest.fixture
def expected_amplitudes():
    return np.array([0, 15, 60, 45, 120])


def test_scale_amplitudes(spike_amplitudes, templates, spike_templates, expected_amplitudes):

    scale_factor = 0.195

    expected = expected_amplitudes * scale_factor
    obtained = write_nwb.scale_amplitudes(spike_amplitudes, templates, spike_templates, scale_factor)

    assert np.allclose(expected, obtained)


def test_read_spike_amplitudes_to_dictionary(tmpdir_factory, spike_amplitudes, templates, spike_templates, expected_amplitudes):
    tmpdir = str(tmpdir_factory.mktemp("spike_amps"))

    spike_amplitudes_path = os.path.join(tmpdir, "spike_amplitudes.npy")
    spike_units_path = os.path.join(tmpdir, "spike_units.npy")
    templates_path = os.path.join(tmpdir, "templates.npy")
    spike_templates_path = os.path.join(tmpdir, "spike_templates.npy")
    inverse_whitening_matrix_path = os.path.join(tmpdir, "inverse_whitening_matrix_path.npy")

    whitening_matrix = np.diag(np.arange(3) + 1)
    inverse_whitening_matrix = np.linalg.inv(whitening_matrix)

    spike_units = np.array([0, 0, 0, 1, 1])

    for idx in range(templates.shape[0]):
        templates[idx, :, :] = np.dot(
            templates[idx, :, :], whitening_matrix
        )

    np.save(spike_amplitudes_path, spike_amplitudes, allow_pickle=False)
    np.save(spike_units_path, spike_units, allow_pickle=False)
    np.save(templates_path, templates, allow_pickle=False)
    np.save(spike_templates_path, spike_templates, allow_pickle=False)
    np.save(inverse_whitening_matrix_path, inverse_whitening_matrix, allow_pickle=False)

    obtained = write_nwb.read_spike_amplitudes_to_dictionary(
        spike_amplitudes_path,
        spike_units_path,
        templates_path,
        spike_templates_path,
        inverse_whitening_matrix_path
    )

    assert np.allclose(expected_amplitudes[:3], obtained[0])
    assert np.allclose(expected_amplitudes[3:], obtained[1])


@pytest.mark.parametrize("spike_times_mapping, spike_amplitudes_mapping, expected", [

    ({12345: np.array([0, 1, 2, -1, 5, 4])},  # spike_times_mapping

     {12345: np.array([0, 1, 2, 3, 4, 5])},  # spike_amplitudes_mapping

     ({12345: np.array([0, 1, 2, 4, 5])},  # expected
      {12345: np.array([0, 1, 2, 5, 4])})),

    ({12345: np.array([0, 1, 2, -1, 5, 4]),  # spike_times_mapping
      54321: np.array([5, 4, 3, -1, 6])},

     {12345: np.array([0, 1, 2, 3, 4, 5]),  # spike_amplitudes_mapping
      54321: np.array([0, 1, 2, 3, 4])},

     ({12345: np.array([0, 1, 2, 4, 5]),  # expected
       54321: np.array([3, 4, 5, 6])},
      {12345: np.array([0, 1, 2, 5, 4]),
       54321: np.array([2, 1, 0, 4])})),
])
def test_filter_and_sort_spikes(spike_times_mapping, spike_amplitudes_mapping, expected):
    expected_spike_times, expected_spike_amplitudes = expected

    obtained_spike_times, obtained_spike_amplitudes = write_nwb.filter_and_sort_spikes(spike_times_mapping,
                                                                                       spike_amplitudes_mapping)

    np.testing.assert_equal(obtained_spike_times, expected_spike_times)
    np.testing.assert_equal(obtained_spike_amplitudes, expected_spike_amplitudes)


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("probes, parsed_probe_data, expected_units_table", [
    ([{"id": 1234,
       "name": "probeA",
       "sampling_rate": 29999.9655245905,
       "lfp_sampling_rate": 2499.99712704921,
       "temporal_subsampling_factor": 2.0,
       "lfp": {},
       "spike_times_path": "/dummy_path",
       "spike_clusters_files": "/dummy_path",
       "mean_waveforms_path": "/dummy_path",
       "channels": [{"id": 1,
                     "probe_id": 1234,
                     "valid_data": True,
                     "local_index": 0,
                     "probe_vertical_position": 10,
                     "probe_horizontal_position": 10,
                     "anterior_posterior_ccf_coordinate": 15.0,
                     "dorsal_ventral_ccf_coordinate": 20.0,
                     "left_right_ccf_coordinate": 25.0,
                     "manual_structure_acronym": "CA1",
                     "impedence": np.nan,
                     "filtering": "Unknown"},
                    {"id": 2,
                     "probe_id": 1234,
                     "valid_data": True,
                     "local_index": 1,
                     "probe_vertical_position": 20,
                     "probe_horizontal_position": 20,
                     "anterior_posterior_ccf_coordinate": 25.0,
                     "dorsal_ventral_ccf_coordinate": 30.0,
                     "left_right_ccf_coordinate": 35.0,
                     "manual_structure_acronym": "CA3",
                     "impedence": np.nan,
                     "filtering": "Unknown"}],

       "units": [{"id": 777,
                  "local_index": 7,
                  "quality": "good",
                  "a": 0.5,
                  "b": 5},
                 {"id": 778,
                  "local_index": 9,
                  "quality": "noise",
                  "a": 1.0,
                  "b": 10}]}],

     (pd.DataFrame({"id": [777, 778], "local_index": [7, 9],  # units_table
                    "a": [0.5, 1.0], "b": [5, 10]}).set_index(keys="id", drop=True),
      {777: np.array([0., 1., 2., -1., 5., 4.]),  # spike_times
       778: np.array([5., 4., 3., -1., 6.])},
      {777: np.array([0., 1., 2., 3., 4., 5.]),  # spike_amplitudes
       778: np.array([0., 1., 2., 3., 4.])},
      {777: np.array([1., 2., 3., 4., 5., 6.]),  # mean_waveforms
       778: np.array([1., 2., 3., 4., 5.])}),

     pd.DataFrame({"id": [777, 778], "local_index": [7, 9],  # units_table
                   "a": [0.5, 1.0], "b": [5, 10],
                   "spike_times": [[0., 1., 2., 4., 5.], [3., 4., 5., 6.]],
                   "spike_amplitudes": [[0., 1., 2., 5., 4.], [2., 1., 0., 4.]],
                   "waveform_mean": [[1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5.]]}
                  ).set_index(keys="id", drop=True)),
])
def test_add_probewise_data_to_nwbfile(monkeypatch, nwbfile, roundtripper,
                                       roundtrip, probes, parsed_probe_data,
                                       expected_units_table):

    def mock_parse_probes_data(probes):
        return parsed_probe_data

    monkeypatch.setattr(write_nwb, "parse_probes_data", mock_parse_probes_data)
    nwbfile = write_nwb.add_probewise_data_to_nwbfile(nwbfile, probes)

    if roundtrip:
        obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(obt.nwbfile.units.to_dataframe(),
                                  expected_units_table)


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("eye_tracking_rig_geometry, expected", [
    ({"monitor_position_mm": [1., 2., 3.],
      "monitor_rotation_deg": [4., 5., 6.],
      "camera_position_mm": [7., 8., 9.],
      "camera_rotation_deg": [10., 11., 12.],
      "led_position": [13., 14., 15.],
      "equipment": "test_rig"},

     #  Expected
     {"geometry": pd.DataFrame({"monitor_position_mm": [1., 2., 3.],
                                "monitor_rotation_deg": [4., 5., 6.],
                                "camera_position_mm": [7., 8., 9.],
                                "camera_rotation_deg": [10., 11., 12.],
                                "led_position_mm": [13., 14., 15.]},
                               index=["x", "y", "z"]),
      "equipment": "test_rig"}),
])
def test_add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile, roundtripper,
                                                       roundtrip,
                                                       eye_tracking_rig_geometry,
                                                       expected):

    nwbfile = write_nwb.add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile,
                                                                      eye_tracking_rig_geometry)
    if roundtrip:
        obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)
    obtained_metadata = obt.get_rig_metadata()

    pd.testing.assert_frame_equal(obtained_metadata["geometry"], expected["geometry"], check_like=True)
    assert obtained_metadata["equipment"] == expected["equipment"]


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize(("eye_tracking_frame_times, eye_dlc_tracking_data, "
                          "eye_gaze_data, expected_pupil_data, expected_gaze_data"), [
    (
        # eye_tracking_frame_times
        pd.Series([3., 4., 5., 6., 7.]),
        # eye_dlc_tracking_data
        {"pupil_params": create_preload_eye_tracking_df(np.full((5, 5), 1.)),
         "cr_params": create_preload_eye_tracking_df(np.full((5, 5), 2.)),
         "eye_params": create_preload_eye_tracking_df(np.full((5, 5), 3.))},
        # eye_gaze_data
        {"raw_pupil_areas": pd.Series([2., 4., 6., 8., 10.]),
         "raw_eye_areas": pd.Series([3., 5., 7., 9., 11.]),
         "raw_screen_coordinates": pd.DataFrame({"y": [2., 4., 6., 8., 10.], "x": [3., 5., 7., 9., 11.]}),
         "raw_screen_coordinates_spherical": pd.DataFrame({"y": [2., 4., 6., 8., 10.], "x": [3., 5., 7., 9., 11.]}),
         "new_pupil_areas": pd.Series([2., 4., np.nan, 8., 10.]),
         "new_eye_areas": pd.Series([3., 5., np.nan, 9., 11.]),
         "new_screen_coordinates": pd.DataFrame({"y": [2., 4., np.nan, 8., 10.], "x": [3., 5., np.nan, 9., 11.]}),
         "new_screen_coordinates_spherical": pd.DataFrame({"y": [2., 4., np.nan, 8., 10.], "x": [3., 5., np.nan, 9., 11.]}),
         "synced_frame_timestamps": pd.Series([3., 4., 5., 6., 7.])},
        # expected_pupil_data
        pd.DataFrame({"corneal_reflection_center_x": [2.] * 5,
                      "corneal_reflection_center_y": [2.] * 5,
                      "corneal_reflection_height": [4.] * 5,
                      "corneal_reflection_width": [4.] * 5,
                      "corneal_reflection_phi": [2.] * 5,
                      "pupil_center_x": [1.] * 5,
                      "pupil_center_y": [1.] * 5,
                      "pupil_height": [2.] * 5,
                      "pupil_width": [2.] * 5,
                      "pupil_phi": [1.] * 5,
                      "eye_center_x": [3.] * 5,
                      "eye_center_y": [3.] * 5,
                      "eye_height": [6.] * 5,
                      "eye_width": [6.] * 5,
                      "eye_phi": [3.] * 5},
                     index=[3., 4., 5., 6., 7.]),
        # expected_gaze_data
        pd.DataFrame({"raw_eye_area": [3., 5., 7., 9., 11.],
                      "raw_pupil_area": [2., 4., 6., 8., 10.],
                      "raw_screen_coordinates_x_cm": [3., 5., 7., 9., 11.],
                      "raw_screen_coordinates_y_cm": [2., 4., 6., 8., 10.],
                      "raw_screen_coordinates_spherical_x_deg": [3., 5., 7., 9., 11.],
                      "raw_screen_coordinates_spherical_y_deg": [2., 4., 6., 8., 10.],
                      "filtered_eye_area": [3., 5., np.nan, 9., 11.],
                      "filtered_pupil_area": [2., 4., np.nan, 8., 10.],
                      "filtered_screen_coordinates_x_cm": [3., 5., np.nan, 9., 11.],
                      "filtered_screen_coordinates_y_cm": [2., 4., np.nan, 8., 10.],
                      "filtered_screen_coordinates_spherical_x_deg": [3., 5., np.nan, 9., 11.],
                      "filtered_screen_coordinates_spherical_y_deg": [2., 4., np.nan, 8., 10.]},
                     index=[3., 4., 5., 6., 7.])
    ),
])
def test_add_eye_tracking_data_to_nwbfile(nwbfile, roundtripper, roundtrip,
                                          eye_tracking_frame_times,
                                          eye_dlc_tracking_data,
                                          eye_gaze_data,
                                          expected_pupil_data, expected_gaze_data):
    nwbfile = write_nwb.add_eye_tracking_data_to_nwbfile(nwbfile,
                                                         eye_tracking_frame_times,
                                                         eye_dlc_tracking_data,
                                                         eye_gaze_data)

    if roundtrip:
        obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)
    obtained_pupil_data = obt.get_pupil_data()
    obtained_screen_gaze_data = obt.get_screen_gaze_data(include_filtered_data=True)

    pd.testing.assert_frame_equal(obtained_pupil_data,
                                  expected_pupil_data, check_like=True)
    pd.testing.assert_frame_equal(obtained_screen_gaze_data,
                                  expected_gaze_data, check_like=True)
