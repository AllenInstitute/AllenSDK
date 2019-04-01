from unittest import mock

import pytest
import numpy as np
import h5py

from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import EcephysSyncDataset


@pytest.mark.parametrize('expected', [1, None])
def test_sample_frequency(expected):
    dataset = EcephysSyncDataset()
    dataset.meta_data = {'ni_daq': {}}

    dataset.sample_frequency = expected
    assert dataset.sample_frequency == expected
    assert dataset.sample_frequency == dataset.meta_data['ni_daq']['counter_output_freq']


@pytest.mark.parametrize('key,line_labels,led_indices,sample_frequency', [
    [ 'foo', ('LED_sync',), np.array([1, 2, 3]), 1000 ],
    [ 'LED_sync', ('LED_sync',), np.array([1, 2, 3]), 1000 ],
])
def test_extract_led_times(key, line_labels, led_indices, sample_frequency):

    dataset = EcephysSyncDataset()
    dataset.line_labels = line_labels
    dataset.sample_frequency = sample_frequency

    with mock.patch('allensdk.brain_observatory.sync_dataset.Dataset.get_rising_edges', return_value=led_indices) as p:
        obtained = dataset.extract_led_times(key)
        
        if key in line_labels:
            p.assert_called_once_with(line_labels.index(key))
        else:
            p.assert_called_once_with(18)

    expected = led_indices / sample_frequency
    assert np.allclose(obtained, expected)

@pytest.mark.parametrize('photodiode_times,vsyncs,cycle,expected', [
    [ # expected timing, using vsyncs
        np.arange(5.0, 5 + (100 * 0.75), 0.75),
        np.arange(5.0, 5 + (298 * 0.25), 0.25) - 0.0625 * np.random.rand(298), # num frames is (num_vsyncs - 1) * cycle + 1 
        3,
        np.arange(5.0, 5 + (298 * 0.25), 0.25)
    ]
])
def test_extract_frame_times_from_photodiode(photodiode_times, vsyncs, cycle, expected):

    class TimesWrapper:
        def __call__(self, ignore, keys):
            if 'photodiode' in keys:
                return photodiode_times
            elif 'frames' in keys:
                return vsyncs

    dataset = EcephysSyncDataset()
    with mock.patch('allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset.EcephysSyncDataset.get_edges', new_callable=TimesWrapper) as p:
        obtained = dataset.extract_frame_times_from_photodiode(photodiode_cycle=cycle)
        assert np.allclose(obtained, expected)



def test_factory():

    with mock.patch('allensdk.brain_observatory.sync_dataset.Dataset.load') as p:
        dataset = EcephysSyncDataset.factory('foo')
        p.assert_called_with('foo')