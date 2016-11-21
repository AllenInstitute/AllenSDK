# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

from mock import patch, MagicMock
from pkg_resources import resource_filename  # @UnresolvedImport
import numpy as np
from allensdk.core.nwb_data_set import NwbDataSet
import pytest
import os

NWB_FLAVORS = []

if 'TEST_EPHYS_NWB_FILES' in os.environ:
    nwb_list_file = os.environ['TEST_EPHYS_NWB_FILES']
else:
    nwb_list_file = resource_filename(__name__, 'nwb_ephys_files.txt')
with open(nwb_list_file, 'r') as f:
    NWB_FLAVORS = [l.strip() for l in f]


@pytest.fixture(params=NWB_FLAVORS)
def data_set(request):
    nwb_file = request.param
    data_set = NwbDataSet(nwb_file)
    return data_set

@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_get_sweep_numbers(data_set):
    sweep_numbers = data_set.get_sweep_numbers()

    assert len(sweep_numbers) > 0


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_get_experiment_sweep_numbers(data_set):
    sweep_numbers = data_set.get_experiment_sweep_numbers()

    assert len(sweep_numbers) > 0


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_get_spike_times(data_set):
    sweep_numbers = data_set.get_experiment_sweep_numbers()

    found_spikes = False

    for n in sweep_numbers:
        spike_times = data_set.get_spike_times(n)
        if len(spike_times > 0):
            found_spikes = True

    assert found_spikes is True

def mock_h5py_file(m=None, data=None):
    if m is None:
        m = MagicMock()

    f = MagicMock()

    if data is None:
        f.__enter__.return_value = f
    else:
        f.__enter__.return_value = data

    m.return_value = f

    return m


@pytest.fixture
def mock_data_set():
    nwb_file = 'fixture.nwb'
    data_set = NwbDataSet(nwb_file)
    return data_set


def test_fill_sweep_responses(mock_data_set):
    data_set = mock_data_set
    DATA_LENGTH = 5

    h5 = {
        'stimulus': {
            'presentation': {
                'Sweep_1': {
                    'aibs_stimulus_amplitude_pa': 15.0,
                    'aibs_stimulus_name': 'Joe',
                    'gain': 1.0,
                    'initial_access_resistance': 0.05,
                    'seal': True
                }
            }
        },
        'epochs': {
            'Sweep_1': {
                'description': 'sweep 1 description',
                'stimulus': {},
                'response': {
                    'count': DATA_LENGTH,
                    'idx_start': 0,
                    'timeseries': {
                        'data': np.ones(DATA_LENGTH) * 1.0
                    }
                }
            }
        }
    }

    with patch('h5py.File', mock_h5py_file(data=h5)):
        data_set.fill_sweep_responses(0.0, [1])

    assert not np.any(h5['epochs']['Sweep_1']['response']['timeseries']['data'])
    assert len(h5['epochs']['Sweep_1']['response']['timeseries']['data']) == \
        DATA_LENGTH


@pytest.mark.xfail
def test_set_spike_times(mock_data_set):
    data_set = mock_data_set
    DATA_LENGTH = 5

    h5 = {
        'analysis': {
            'spike_times': {
                'Sweep_1': {}
            }
        },
        'stimulus': {
            'presentation': {
                'Sweep_1': {
                    'aibs_stimulus_amplitude_pa': 15.0,
                    'aibs_stimulus_name': 'Joe',
                    'gain': 1.0,
                    'initial_access_resistance': 0.05,
                    'seal': True
                }
            }
        },
        'epochs': {
            'Sweep_1': {
                'description': 'sweep 1 description',
                'stimulus': {},
                'response': {
                    'count': DATA_LENGTH,
                    'idx_start': 0,
                    'timeseries': {
                        'data': np.ones(DATA_LENGTH) * 1.0
                    }
                }
            }
        }
    }

    with patch('h5py.File', mock_h5py_file(data=h5)):
        data_set.set_spike_times(1, [0.1, 0.2, 0.3, 0.4, 0.5])

    assert False

@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_get_sweep_metadata(data_set):
    sweep_metadata = data_set.get_sweep_metadata(1)

    assert sweep_metadata is not None
