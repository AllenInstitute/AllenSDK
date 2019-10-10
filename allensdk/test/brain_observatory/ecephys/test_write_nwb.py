import os
from datetime import datetime, timezone
from pathlib import Path
import logging

import pytest
import pynwb
import pandas as pd
import numpy as np

from pynwb import NWBFile, NWBHDF5IO

from allensdk.brain_observatory.ecephys.current_source_density.__main__ import write_csd_to_h5
import allensdk.brain_observatory.ecephys.write_nwb.__main__ as write_nwb
from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysNwbSessionApi


@pytest.fixture
def units_table():
    return pynwb.misc.Units.from_dataframe(pd.DataFrame({
        'peak_channel_id': [5, 10, 15],
        'local_index':[0, 1, 2],
        'quality': ['good', 'good', 'noise'],
        'firing_rate': [0.5, 1.2, -3.14],
        'snr': [1.0, 2.4, 5],
        'isi_violations': [34, 39, 22]
    }, index=pd.Index([11, 22, 33], name='id')), name='units')


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
        "net_rotation": [-np.pi, -2 *np.pi, -np.pi, 0, np.pi]
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
        "color": ["1.0", "", r"[1.0,-1.0, 10., -42.12, -.1]", "-1.0", ""]
    }, index=pd.Index(name='stimulus_presentations_id', data=[0, 1, 2, 3, 4]))


def test_roundtrip_basic_metadata(roundtripper):
    dt = datetime.now(timezone.utc)
    nwbfile = pynwb.NWBFile(
        session_description='EcephysSession',
        identifier='{}'.format(12345),
        session_start_time=dt
    )

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    assert 12345 == api.get_ecephys_session_id()
    assert dt == api.get_session_start_time()


def test_add_metadata(nwbfile, roundtripper):
    metadata = {
      "specimen_name": "mouse_1",
      "age_in_days": 100.0,
      "full_genotype": "wt",
      "strain": "c57",
      "sex": "F",
      "stimulus_name": "brain_observatory_2.0"
    }
    write_nwb.add_metadata_to_nwbfile(nwbfile, metadata)

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained = api.get_metadata()

    assert set(metadata.keys()) == set(obtained.keys())

    misses = {}
    for key, value in metadata.items():
        if obtained[key] != value:
            misses[key] = {"expected": value, "obtained": obtained[key]}

    assert len(misses) == 0, f"the following metadata items were mismatched: {misses}"


def test_add_stimulus_presentations(nwbfile, stimulus_presentations, roundtripper):
    write_nwb.add_stimulus_timestamps(nwbfile, [0, 1])
    write_nwb.add_stimulus_presentations(nwbfile, stimulus_presentations)

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained_stimulus_table = api.get_stimulus_presentations()
    
    pd.testing.assert_frame_equal(stimulus_presentations, obtained_stimulus_table, check_dtype=False)


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


def test_add_optotagging_table_to_nwbfile(nwbfile, roundtripper):
    opto_table = pd.DataFrame({
        "start_time": [0., 1., 2., 3.],
        "stop_time": [0.5, 1.5, 2.5, 3.5],
        "level": [10., 9., 8., 7.],
        "condition": ["a", "a", "b", "c"]
    })
    opto_table["duration"] = opto_table["stop_time"] - opto_table["start_time"]

    nwbfile = write_nwb.add_optotagging_table_to_nwbfile(nwbfile, opto_table)
    api = roundtripper(nwbfile, EcephysNwbSessionApi)

    obtained = api.get_optogenetic_stimulation()
    pd.set_option("display.max_columns", None)
    print(obtained)
    
    pd.testing.assert_frame_equal(opto_table, obtained, check_like=True)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize('pid,desc,srate,lfp_srate,has_lfp,expected', [
    [
        12, 
        'a probe', 
        30000.0,
        2500.0,
        True,
        pd.DataFrame({
            'description': ['a probe'], 
            'sampling_rate': [30000.0], 
            "lfp_sampling_rate": [2500.0],
            "has_lfp_data": [True],
            "location": [""]
        }, index=pd.Index([12], name='id'))
    ]
])
def test_add_probe_to_nwbfile(nwbfile, roundtripper, roundtrip, pid, desc, srate, lfp_srate, has_lfp,expected):

    nwbfile, _, _ = write_nwb.add_probe_to_nwbfile(nwbfile, pid,
                                                   description=desc,
                                                   sampling_rate=srate,
                                                   lfp_sampling_rate=lfp_srate,
                                                   has_lfp_data=has_lfp)
    if roundtrip:
        obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(expected, obt.get_probes(), check_like=True)


def test_prepare_probewise_channel_table():

    channels = [
        {
            'id': 2,
            'probe_id': 12,
            'local_index': 44,
            'probe_vertical_position': 21,
            'probe_horizontal_position': 30
        },
        {
            'id': 1,
            'probe_id': 12,
            'local_index': 43,
            'probe_vertical_position': 20,
            'probe_horizontal_position': 30
        }
    ]

    dev = pynwb.device.Device(name='foo')
    eg = pynwb.ecephys.ElectrodeGroup(name='foo_group', description='', location='', device=dev)

    expected = pd.DataFrame({
        'probe_id': [12, 12],
        'local_index': [44, 43],
        'probe_vertical_position': [21, 20],
        'probe_horizontal_position': [30, 30],
        'group': [eg] * 2
    }, index=pd.Index([2, 1], name='id'))

    obtained = write_nwb.prepare_probewise_channel_table(channels, eg)

    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize('dc,order,exp_idx,exp_data', [
    [{'a': [1, 2, 3], 'b': [4, 5, 6]}, ['a', 'b'], [3, 6], [1, 2, 3, 4, 5, 6]]
])
def test_dict_to_indexed_array(dc, order, exp_idx, exp_data):

    obt_idx, obt_data = write_nwb.dict_to_indexed_array(dc, order)
    assert np.allclose(exp_idx, obt_idx)
    assert np.allclose(exp_data, obt_data)
    

def test_add_ragged_data_to_dynamic_table(units_table, spike_times):

    write_nwb.add_ragged_data_to_dynamic_table(
        table=units_table,
        data=spike_times,
        column_name='spike_times'
    )

    assert np.allclose([1, 2, 3, 4, 5, 6], units_table['spike_times'][0])
    assert np.allclose([], units_table['spike_times'][1])
    assert np.allclose([13, 4, 12], units_table['spike_times'][2])


@pytest.mark.parametrize('roundtrip,include_rotation', [
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


@pytest.mark.parametrize('roundtrip', [[True]])
def test_add_raw_running_data_to_nwbfile(nwbfile, raw_running_data, roundtripper, roundtrip):

    nwbfile = write_nwb.add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)
    if roundtrip:
        api_obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        api_obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    obtained = api_obt.get_raw_running_data()

    expected = raw_running_data.rename(columns={"dx": "net_rotation", "vsig": "signal_voltage", "vin": "supply_voltage"})
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


def test_read_stimulus_table(tmpdir_factory, stimulus_presentations):
    dirname = str(tmpdir_factory.mktemp('ecephys_nwb_test'))
    stim_table_path = os.path.join(dirname, 'stim_table.csv')

    stimulus_presentations.to_csv(stim_table_path)
    obt = write_nwb.read_stimulus_table(stim_table_path, column_renames_map={'alpha': 'beta'})

    assert np.allclose(stimulus_presentations['alpha'].values, obt['beta'].values)


# read_spike_times_to_dictionary(spike_times_path, spike_units_path, local_to_global_unit_map=None)
def test_read_spike_times_to_dictionary(tmpdir_factory):
    dirname = str(tmpdir_factory.mktemp('ecephys_nwb_spike_times'))
    spike_times_path = os.path.join(dirname, 'spike_times.npy')
    spike_units_path = os.path.join(dirname, 'spike_units.npy')

    spike_times = np.sort(np.random.rand(30))
    np.save(spike_times_path, spike_times, allow_pickle=False)

    spike_units = np.concatenate([np.arange(15), np.arange(15)])
    np.save(spike_units_path, spike_units, allow_pickle=False)

    local_to_global_unit_map = {ii: -ii for ii in spike_units}

    obtained = write_nwb.read_spike_times_to_dictionary(spike_times_path, spike_units_path, local_to_global_unit_map)
    for ii in range(15):
        assert np.allclose(obtained[-ii], sorted([spike_times[ii], spike_times[15 + ii]]))


def test_read_waveforms_to_dictionary(tmpdir_factory):
    dirname = str(tmpdir_factory.mktemp('ecephys_nwb_mean_waveforms'))
    waveforms_path = os.path.join(dirname, 'mean_waveforms.npy')

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
        'data': np.arange(total_timestamps * len(subsample_channels), dtype=np.int16).reshape((total_timestamps, len(subsample_channels))),
        'timestamps': np.linspace(0, 1, total_timestamps),
        'subsample_channels': subsample_channels
    }


def test_write_probe_lfp_file(tmpdir_factory, lfp_data):

    tmpdir = Path(tmpdir_factory.mktemp("probe_lfp_nwb"))
    input_data_path = tmpdir / Path("lfp_data.dat")
    input_timestamps_path = tmpdir / Path("lfp_timestamps.npy")
    input_channels_path = tmpdir / Path("lfp_channels.npy")
    input_csd_path = tmpdir / Path("csd.h5")
    output_path = str(tmpdir / Path("lfp.nwb"))  # pynwb.NWBHDF5IO chokes on Path

    probe_data = {
        "id": 12345,
        "name": "probeA",
        "sampling_rate": 29.0,
        "lfp_sampling_rate": 10.0,
        "temporal_subsampling_factor": 2.0,
        "channels":  [
            {
                'id': 0,
                'probe_id': 12,
                'local_index': 1,
                'probe_vertical_position': 21,
                'probe_horizontal_position': 33
            },
            {
                'id': 1,
                'probe_id': 12,
                'local_index': 2,
                'probe_vertical_position': 21,
                'probe_horizontal_position': 32
            },
            {
                'id': 2,
                'probe_id': 12,
                'local_index': 3,
                'probe_vertical_position': 21,
                'probe_horizontal_position': 31
            }
        ],
        "lfp": {
            "input_data_path": input_data_path,
            "input_timestamps_path": input_timestamps_path,
            "input_channels_path": input_channels_path,
            "output_path": output_path
        },
        "csd_path": input_csd_path,
        "amplitude_scale_factor": 1.0
    }

    csd = np.arange(20).reshape([2, 10])
    csd_times = np.linspace(-1, 1, 10)
    csd_channels = np.array([3, 2])
    csd_locations = np.array([[1, 2], [3, 3]])

    write_csd_to_h5(
        path=input_csd_path,
        csd=csd,
        relative_window=csd_times,
        channels=csd_channels,
        csd_locations=csd_locations,
        stimulus_name="foo",
        stimulus_index=None,
        num_trials=1000
    )

    np.save(input_timestamps_path, lfp_data["timestamps"],  allow_pickle=False)
    np.save(input_channels_path, lfp_data["subsample_channels"], allow_pickle=False)
    with open(input_data_path, "wb") as input_data_file:
        input_data_file.write(lfp_data["data"].tobytes())

    write_nwb.write_probe_lfp_file(datetime.now(), logging.INFO, probe_data)

    exp_electrodes = pd.DataFrame(probe_data["channels"]).set_index("id").loc[[2, 1], :]

    with pynwb.NWBHDF5IO(output_path, "r") as obt_io:
        obt_f = obt_io.read()

        obt_ser = obt_f.get_acquisition("probe_12345_lfp").electrical_series["probe_12345_lfp_data"]
        assert np.allclose(lfp_data["data"], obt_ser.data[:])
        assert np.allclose(lfp_data["timestamps"], obt_ser.timestamps[:])

        obt_electrodes = obt_f.electrodes.to_dataframe().loc[
            :, ['local_index', 'probe_horizontal_position', 'probe_id','probe_vertical_position']
        ]

        pd.testing.assert_frame_equal(exp_electrodes, obt_electrodes, check_like=True)

        csd_series = obt_f.get_processing_module("current_source_density")["current_source_density"]

        assert np.allclose(csd, csd_series.data[:])
        assert np.allclose(csd_times, csd_series.timestamps[:])
        assert np.allclose([[1, 2], [3, 3]], csd_series.control[:])  # csd interpolated channel locations

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
        session_description='EcephysSession',
        identifier='{}'.format(739448407),
        session_start_time=datetime.now()
    )

    nwbfile = write_nwb.add_invalid_times(nwbfile, invalid_epochs)

    with NWBHDF5IO(nwbfile_name, mode='w') as io:
        io.write(nwbfile)
    nwbfile_in = NWBHDF5IO(nwbfile_name, mode='r').read()

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

    assert s['start_time'] == epoch['start_time']
    assert s['stop_time'] == epoch['end_time']
    assert s['tags'] == [epoch['type'], str(epoch['id']), epoch['label']]


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