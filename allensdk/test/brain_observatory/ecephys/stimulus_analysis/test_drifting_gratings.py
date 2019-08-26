import numpy as np
import pandas as pd
import pytest

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.drifting_gratings import DriftingGratings


pd.set_option('display.max_columns', None)


class MockDGSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        features = np.array(np.meshgrid([1.0, 2.0, 4.0, 8.0, 15.0],                            # TF
                                        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])  # ORI
                            ).reshape(2, 40)

        return pd.DataFrame({
            'start_time': np.concatenate(([0.0], np.linspace(0.5, 78.5, 40, endpoint=True), [80.0])),
            'stop_time': np.concatenate(([0.0], np.linspace(2.5, 80.5, 40, endpoint=True), [81.0])),
            'stimulus_name': ['spontaneous'] + ['drifting_gratings']*40 + ['spontaneous'],
            'stimulus_block': [0] + [1]*40 + [0],
            'duration': [0.5] + [2.0]*40 + [0.5],
            'stimulus_index': [0] + [1]*40 + [0],
            'temporal_frequency': np.concatenate(([np.nan], features[0, :], [np.nan])),
            'orientation': np.concatenate(([np.nan], features[1, :], [np.nan])),
            'contrast': 0.8
        }, index=pd.Index(name='id', data=np.arange(42)))


@pytest.fixture
def ecephys_api():
    return MockDGSessionApi()


def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    dg = DriftingGratings(ecephys_session=session)
    assert(dg.name == 'Drifting Gratings')
    assert(set(dg.unit_ids) == set(range(6)))
    assert(len(dg.conditionwise_statistics) == 40*6)
    assert(dg.conditionwise_psth.shape == (40, 2.0/0.001-1, 6))
    assert(not dg.presentationwise_spike_times.empty)
    assert(len(dg.presentationwise_statistics) == 40*6)
    assert(len(dg.stimulus_conditions) == 40)


def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    dg = DriftingGratings(ecephys_session=session)
    assert(isinstance(dg.stim_table, pd.DataFrame))
    assert(len(dg.stim_table) == 40)
    assert(set(dg.stim_table.columns).issuperset({'temporal_frequency', 'orientation', 'contrast', 'start_time',
                                                  'stop_time'}))

    assert(set(dg.tfvals) == {1.0, 2.0, 4.0, 8.0, 15.0})
    assert(dg.number_tf == 5)

    assert(set(dg.orivals) == {0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0})
    assert(dg.number_ori == 8)

    assert(set(dg.contrastvals) == {0.8})
    assert(dg.number_contrast == 1)


def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    dg = DriftingGratings(ecephys_session=session)
    assert(isinstance(dg.metrics, pd.DataFrame))
    assert(len(dg.metrics) == 6)
    assert(dg.metrics.index.names == ['unit_id'])

    assert('pref_ori_dg' in dg.metrics.columns)
    assert(np.all(dg.metrics['pref_ori_dg'].loc[[0, 1, 2, 3, 4, 5]] == np.full(6, 0.0)))

    assert('pref_tf_dg' in dg.metrics.columns)
    assert(np.all(dg.metrics['pref_tf_dg'].loc[[0, 5]] == [1.0, 2.0]))

    assert('f1_f0_dg' in dg.metrics.columns)
    assert('mod_idx_dg' in dg.metrics.columns)
    assert('g_osi_dg' in dg.metrics.columns)
    assert('g_dsi_dg' in dg.metrics.columns)
    assert('firing_rate_dg' in dg.metrics.columns)
    assert('reliability_dg' in dg.metrics.columns)
    assert('fano_dg' in dg.metrics.columns)
    assert('lifetime_sparseness_dg' in dg.metrics.columns)
    assert('run_pval_dg' in dg.metrics.columns)
    assert('run_mod_dg' in dg.metrics.columns)


@pytest.mark.skip(reason='Function is broken')
def test_contrast_curve():
    pass


@pytest.mark.skip(reason='Function is broken')
def test_c50():
    pass


@pytest.mark.skip(reason='Function is broken')
def test_f1_f0():
    pass


@pytest.mark.skip(reason='Function is broken')
def test_modulation_index():
    pass


if __name__ == '__main__':
    # test_stimulus()
    test_metrics()
