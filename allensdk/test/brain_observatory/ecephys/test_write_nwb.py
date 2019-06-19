import os
from datetime import datetime
from pathlib import Path
import logging

import pytest
import pynwb
import pandas as pd
import numpy as np

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
        "rotation": [-np.pi, -2 *np.pi, -np.pi, 0, np.pi]
    })


def test_roundtrip_metadata(roundtripper):
    nwbfile = pynwb.NWBFile(
        session_description='EcephysSession',
        identifier='{}'.format(12345),
        session_start_time=datetime.now()
    )

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    assert 12345 == api.get_ecephys_session_id()


def test_add_stimulus_presentations(nwbfile, stimulus_presentations, roundtripper):
    write_nwb.add_stimulus_timestamps(nwbfile, [0, 1])
    write_nwb.add_stimulus_presentations(nwbfile, stimulus_presentations)

    api = roundtripper(nwbfile, EcephysNwbSessionApi)
    obtained_stimulus_table = api.get_stimulus_presentations()
    
    pd.testing.assert_frame_equal(stimulus_presentations, obtained_stimulus_table, check_dtype=False)
    

@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize('pid,desc,loc, expected', [
    [12, 'a probe', 'probeA', pd.DataFrame({'description': ['a probe'], 'location': ['probeA'], 'sampling_rate': [30000.0]}, index=pd.Index([12], name='id'))]
])
def test_add_probe_to_nwbfile(nwbfile, roundtripper, roundtrip, pid, desc, loc, expected):

    nwbfile, _, _ = write_nwb.add_probe_to_nwbfile(nwbfile, pid, description=desc, location=loc)
    if roundtrip:
        obt = roundtripper(nwbfile, EcephysNwbSessionApi)
    else:
        obt = EcephysNwbSessionApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(expected, obt.get_probes())


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
        expected = expected.drop(columns="rotation")
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

    spike_times = np.random.rand(30)
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
    output_path = str(tmpdir / Path("lfp.nwb"))  # pynwb.NWBHDF5IO chokes on Path

    probe_data = {
        "id": 12345,
        "name": "probeA",
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
        }
    }

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