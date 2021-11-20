import pytest
import pandas as pd
import numpy as np

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.stimulus_analysis.flashes import Flashes
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


class MockFlSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        return pd.DataFrame({
            'start_time': np.concatenate(([0.0], np.linspace(0.5, 4.25, 16, endpoint=True), [4.5])),
            'stop_time': np.concatenate(([0.5], np.linspace(0.75, 4.5, 16, endpoint=True), [5.0])),
            'stimulus_name': ['spontaneous'] + ['flashes']*16 + ['spontaneous'],
            'stimulus_block': [0] + [1]*16 + [0],
            'duration': [0.5] + [0.25]*16 + [0.5],
            'stimulus_index': [0] + [1]*16 + [0],
            'color': [np.nan, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, np.nan]
        }, index=pd.Index(name='id', data=np.arange(18)))

    def get_invalid_times(self):
        return pd.DataFrame()



@pytest.fixture
def ecephys_api():
    return MockFlSessionApi()


def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    fl = Flashes(ecephys_session=session)
    assert(fl.name == 'Flashes')
    assert(set(fl.unit_ids) == set(range(6)))
    assert(len(fl.conditionwise_statistics) == 2*6)
    assert(fl.conditionwise_psth.shape == (2, 249, 6))
    assert(not fl.presentationwise_spike_times.empty)
    assert(len(fl.presentationwise_statistics) == 16*6)
    assert(len(fl.stimulus_conditions) == 2)


def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    fl = Flashes(ecephys_session=session)
    assert(isinstance(fl.stim_table, pd.DataFrame))
    assert(len(fl.stim_table) == 16)
    assert(set(fl.stim_table.columns).issuperset({'color', 'start_time', 'stop_time'}))

    assert(all(fl.colors == [-1.0, 1.0]))
    assert(fl.number_colors == 2)


def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    fl = Flashes(ecephys_session=session)
    assert(isinstance(fl.metrics, pd.DataFrame))
    assert(len(fl.metrics) == 6)
    assert(fl.metrics.index.names == ['unit_id'])

    assert('on_off_ratio_fl' in fl.metrics.columns)
    assert(np.allclose(fl.metrics['on_off_ratio_fl'].loc[[0, 1, 2, 3, 4, 5]],
                       [0.0, np.nan, 0.0, np.nan, 3.0, 2.0], equal_nan=True))  # Check _get_on_off_ratio() method

    assert('sustained_idx_fl' in fl.metrics.columns)
    assert(np.allclose(fl.metrics['sustained_idx_fl'].loc[[0, 1, 2, 3, 4, 5]].values,
                       [0.00401606, np.nan, 0.01204819, np.nan, 0.02811245, 0.00401606], equal_nan=True))

    assert('firing_rate_fl' in fl.metrics.columns)
    assert('time_to_peak_fl' in fl.metrics.columns)
    assert('fano_fl' in fl.metrics.columns)
    assert('lifetime_sparseness_fl' in fl.metrics.columns)
    assert('run_pval_fl' in fl.metrics.columns)
    assert('run_mod_fl' in fl.metrics.columns)


if __name__ == '__main__':
    # test_load()
    # test_stimulus()
    test_metrics()
