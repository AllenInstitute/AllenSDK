import pytest
import numpy as np
from mock import Mock
import allensdk.internal.model.biophysical.ephys_utils as ephys_utils


@pytest.fixture
def data_set():
    data = { 'stimulus': 1.0 * np.arange(10),
             'response': 1.0 * np.arange(10),
             'sampling_rate': 0.1
            }
    
    data_set = Mock()
    data_set.get_sweep = Mock(name='sweep_data',
                              return_value=data)
    
    return data_set


def test_passive_preprocess(data_set):
    s = { 'sweep_number': 5 }

    v, i, t = ephys_utils.get_sweep_v_i_t_from_set(data_set,
                                                   s['sweep_number'])
    assert np.array_equal(v, np.arange(10) * 1000.0)
    assert np.array_equal(i, np.arange(10) * 1.0e12)
    assert np.array_equal(t, np.arange(10) * 10.0)
