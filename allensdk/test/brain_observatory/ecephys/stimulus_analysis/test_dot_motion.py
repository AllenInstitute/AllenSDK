import pytest
import numpy as np
import pandas as pd

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.stimulus_analysis.dot_motion import DotMotion
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


class MockDMSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        features = np.array(np.meshgrid([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],  # Dir
                                        [0.001, 0.005, 0.01, 0.02])                            # Speed
                            ).reshape(2, 32)

        features = np.concatenate((features, np.array([np.nan, np.nan]).reshape((2, 1))), axis=1) # null case

        return pd.DataFrame({
            'start_time': np.concatenate(([0.0], np.linspace(0.5, 32.5, 33, endpoint=True), [33.5])),
            'stop_time': np.concatenate(([0.5], np.linspace(1.5, 33.5, 33, endpoint=True), [34.0])),
            'stimulus_name': ['spontaneous'] + ['dot_motion']*33 + ['spontaneous'],
            'stimulus_block': [0] + [1]*33 + [0],
            'duration': [0.5] + [1.0]*33 + [0.5],
            'stimulus_index': [0] + [1]*33 + [0],
            'Dir': np.concatenate(([np.nan], features[0,:], [np.nan])),
            'Speed': np.concatenate(([np.nan], features[1, :], [np.nan]))
        }, index=pd.Index(name='id', data=np.arange(35)))

    def get_invalid_times(self):
        return pd.DataFrame()



@pytest.fixture
def ecephys_api():
    return MockDMSessionApi()


def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    dm = DotMotion(ecephys_session=session)
    assert(dm.name == 'Dot Motion')
    assert(set(dm.unit_ids) == set(range(6)))
    assert(len(dm.conditionwise_statistics) == 33*6)
    assert(dm.conditionwise_psth.shape == (33, 1.0/0.001-1, 6))
    assert(not dm.presentationwise_spike_times.empty)
    assert(len(dm.presentationwise_statistics) == 33*6)
    assert(len(dm.stimulus_conditions) == 33)


def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    dm = DotMotion(ecephys_session=session)
    assert(isinstance(dm.stim_table, pd.DataFrame))
    assert(len(dm.stim_table) == 33)
    assert(set(dm.stim_table.columns).issuperset({'Dir', 'Speed', 'start_time', 'stop_time'}))

    assert(set(dm.directions) == {0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0})
    assert(dm.number_directions == 8)

    assert(set(dm.speeds) == {0.001, 0.005, 0.01, 0.02})
    assert(dm.number_speeds == 4)


def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    rfm = DotMotion(ecephys_session=session)
    assert(isinstance(rfm.metrics, pd.DataFrame))
    assert(len(rfm.metrics) == 6)
    assert(rfm.metrics.index.names == ['unit_id'])

    assert('pref_speed_dm' in rfm.metrics.columns)
    assert(rfm.metrics['pref_speed_dm'].loc[0] == 0.001)
    assert(rfm.metrics['pref_speed_dm'].loc[5] == 0.001)

    assert('pref_dir_dm' in rfm.metrics.columns)
    assert(rfm.metrics['pref_dir_dm'].loc[0] == 0.0)
    assert(rfm.metrics['pref_dir_dm'].loc[4] == 45.0)

    assert('firing_rate_dm' in rfm.metrics.columns)
    assert('fano_dm' in rfm.metrics.columns)
    assert('lifetime_sparseness_dm' in rfm.metrics.columns)
    assert('run_pval_dm' in rfm.metrics.columns)
    assert('run_mod_dm' in rfm.metrics.columns)


@pytest.mark.skip(reason='metric not yet implemented')
def test_speed_tuning_idx():
    pass


if __name__ == '__main__':
    # test_load()
    # test_stimulus()
    test_metrics()
