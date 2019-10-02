# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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

@pytest.mark.nightly
def test_get_sweep_numbers(data_set):
    sweep_numbers = data_set.get_sweep_numbers()

    assert len(sweep_numbers) > 0


@pytest.mark.nightly
def test_get_experiment_sweep_numbers(data_set):
    sweep_numbers = data_set.get_experiment_sweep_numbers()

    assert len(sweep_numbers) > 0


@pytest.mark.nightly
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


def test_fill_sweep_responses_extend(mock_data_set):
    data_set = mock_data_set
    DATA_LENGTH = 5

    class H5Scalar(object):
        def __init__(self, i):
            self.i = i
            self.value = i
        def __eq__(self, j):
            return j == self.i
        
    h5 = {
        'epochs': {
            'Sweep_1': {
                'response': {
                    'timeseries': {
                        'data': np.ones(DATA_LENGTH)
                    }
                }
            },
            'Experiment_1': {
                'stimulus': {
                    'idx_start': H5Scalar(1),
                    'count': H5Scalar(3), # truncation is here
                    'timeseries': {
                        'data': np.ones(DATA_LENGTH)
                        }
                    }
                }
        }
        }

    with patch('h5py.File', mock_h5py_file(data=h5)):
        data_set.fill_sweep_responses(0.0, [1], extend_experiment=True)

    assert h5['epochs']['Experiment_1']['stimulus']['count'] == 4
    assert h5['epochs']['Experiment_1']['stimulus']['idx_start'] == 1
    assert np.all(h5['epochs']['Sweep_1']['response']['timeseries']['data']== 0.0)

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

@pytest.mark.nightly
def test_get_sweep_metadata(data_set):
    sweep_metadata = data_set.get_sweep_metadata(1)

    assert sweep_metadata is not None
