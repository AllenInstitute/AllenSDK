import os
from datetime import datetime
import warnings

import pytest
import pynwb
import pandas as pd
import numpy as np

import allensdk.brain_observatory.ecephys.write_nwb.__main__ as write_nwb
from allensdk.brain_observatory.ecephys.ecephys_api.ecephys_nwb_api import EcephysNwbApi
from allensdk.brain_observatory.ecephys import RunningSpeed


@pytest.fixture
def nwbfile():
    return pynwb.NWBFile(
        session_description='asession',
        identifier='afile',
        session_start_time=datetime.now()
    )


@pytest.fixture
def roundtripper(tmpdir_factory):
    def f(nwbfile):
        tmpdir = str(tmpdir_factory.mktemp('ecephys_nwb_roundtrip_tests'))
        nwb_path = os.path.join(tmpdir, 'nwbfile.nwb')

        with pynwb.NWBHDF5IO(nwb_path, 'w') as write_io:
            write_io.write(nwbfile)

        return EcephysNwbApi(nwb_path)
    return f


@pytest.fixture
def stimulus_table_data():
    return pd.DataFrame({
        'start_time': [1, 2, 4, 5, 6],
        'stop_time': [2, 4, 5, 6, 8],
        'alpha': [0.5, 0.4, 0.3, 0.2, 0.1]
    })


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
    return RunningSpeed(
        timestamps=[1, 2, 3],
        values=[4, 5, 6]
    )


def test_add_stimulus_table_to_file(nwbfile, stimulus_table_data, roundtripper):
    write_nwb.add_stimulus_table_to_file(nwbfile, stimulus_table_data)

    api = roundtripper(nwbfile)
    obtained_stimulus_table = api.get_stimulus_table()
    
    print(obtained_stimulus_table)
    print(stimulus_table_data)

    pd.testing.assert_frame_equal(stimulus_table_data, obtained_stimulus_table, check_dtype=False)
    

@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize('pid,desc,loc, expected', [
    [12, 'a probe', 'probeA', pd.DataFrame({'description': ['a probe'], 'location': ['probeA']}, index=pd.Index([12], name='id'))]
])
def test_add_probe_to_nwbfile(nwbfile, roundtripper, roundtrip, pid, desc, loc, expected):

    nwbfile, _, _ = write_nwb.add_probe_to_nwbfile(nwbfile, pid, description=desc, location=loc)
    if roundtrip:
        obt = roundtripper(nwbfile)
    else:
        obt = EcephysNwbApi.from_nwbfile(nwbfile)

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

    dev  = pynwb.device.Device(name='foo')
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


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_speed_to_nwbfile(nwbfile, running_speed, roundtripper, roundtrip):

    nwbfile = write_nwb.add_running_speed_to_nwbfile(nwbfile, running_speed)
    if roundtrip:
        api_obt = roundtripper(nwbfile)
    else:
        api_obt = EcephysNwbApi.from_nwbfile(nwbfile)

    running_speed_obt = api_obt.get_running_speed()
    assert np.allclose(running_speed.timestamps, running_speed_obt.timestamps)
    assert np.allclose(running_speed.values, running_speed_obt.values)


def test_read_stimulus_table(tmpdir_factory, stimulus_table_data):
    dirname = str(tmpdir_factory.mktemp('ecephys_nwb_test'))
    stim_table_path = os.path.join(dirname, 'stim_table.csv')

    stimulus_table_data.to_csv(stim_table_path)
    obt = write_nwb.read_stimulus_table(stim_table_path, column_renames_map={'alpha': 'beta'})

    assert np.allclose(stimulus_table_data['alpha'].values, obt['beta'].values)


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
        assert np.allclose(obtained[-ii], sorted([spike_times[ii], spike_times[15+ii]]))


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
