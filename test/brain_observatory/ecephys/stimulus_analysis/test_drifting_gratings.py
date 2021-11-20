import numpy as np
import pandas as pd
import pytest

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.drifting_gratings import DriftingGratings, modulation_index, c50, f1_f0


pd.set_option('display.max_columns', None)


class MockDGSessionApi(MockSessionApi):
    ## c50 will be calculated differently depending on if 'drifting_gratings_contrast' stimuli exists.

    def __init__(self, with_dg_contrast=False):
        self._with_dg_contrast = with_dg_contrast

    def get_spike_times(self):
        return {
            0: np.array([1, 2, 3, 4]),
            1: np.array([2.5]),
            2: np.array([1.01, 1.03, 1.02]),
            3: np.array([]),
            4: np.array([0.01, 1.7, 2.13, 3.19, 4.25, 46.4, 48.7, 54.2, 80.3, 85.40, 85.44, 85.47]),
            #5: np.array([1.5, 3.0, 4.5, 90.1])  # make sure there is a spike for the contrast stimulus
            5: np.concatenate(([1.5, 3.0, 4.5], np.linspace(85.0, 89.0, 20)))
        }

    def get_stimulus_presentations(self):
        features = np.array(np.meshgrid([1.0, 2.0, 4.0, 8.0, 15.0],                            # TF
                                        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])  # ORI
                            ).reshape(2, 40)

        stim_table = pd.DataFrame({
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

        if self._with_dg_contrast:
            features = np.array(np.meshgrid([0.0, 45.0, 90.0, 135.0],                             # ORI
                                            [0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0])  # contrast
                                ).reshape(2, 36)

            dg_constrast = pd.DataFrame({
                'start_time': np.concatenate((80.0 + np.linspace(0.0, 17.5, 36, endpoint=True), [97.5])),
                'stop_time': np.concatenate((81.5 + np.linspace(0.5, 18.0, 36, endpoint=True), [98.0])),
                'stimulus_name': ['drifting_gratings_contrast']*36 + ['spontaneous'],
                'stimulus_block': [2]*36 + [0],
                'duration': [0.5]*36 + [0.5],
                'stimulus_index': [2]*36 + [0],
                'temporal_frequency': 2.0,
                'orientation': np.concatenate((features[0, :], [np.nan])),
                'contrast': np.concatenate((features[1, :], [np.nan]))
            }, index=pd.Index(name='id', data=np.arange(42, 42+37)))
            stim_table = pd.concat((stim_table, dg_constrast))

        return stim_table

    def get_invalid_times(self):
        return pd.DataFrame()




@pytest.fixture
def ecephys_api():
    return MockDGSessionApi()

#def mock_ecephys_api():
#    return MockDGSessionApi()

@pytest.fixture
def ecephys_api_w_contrast():
    return MockDGSessionApi(with_dg_contrast=True)



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
    assert(len(dg.stim_table_contrast) == 0)

    assert(set(dg.stim_table.columns).issuperset({'temporal_frequency', 'orientation', 'contrast', 'start_time',
                                                  'stop_time'}))

    assert(set(dg.tfvals) == {1.0, 2.0, 4.0, 8.0, 15.0})
    assert(dg.number_tf == 5)

    assert(set(dg.orivals) == {0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0})
    assert(dg.number_ori == 8)

    assert(set(dg.contrastvals) == {0.8})
    assert(dg.number_contrast == 1)


def test_metrics(ecephys_api):
    # Run metrics with no drifting_gratings_contrast stimuli
    session = EcephysSession(api=ecephys_api)
    dg = DriftingGratings(ecephys_session=session)
    assert(isinstance(dg.metrics, pd.DataFrame))
    assert(len(dg.metrics) == 6)
    assert(dg.metrics.index.names == ['unit_id'])

    assert('pref_ori_dg' in dg.metrics.columns)
    assert(np.all(dg.metrics['pref_ori_dg'].loc[[0, 1, 2, 3, 4, 5]] == np.full(6, 0.0)))

    assert('pref_tf_dg' in dg.metrics.columns)
    assert(np.all(dg.metrics['pref_tf_dg'].loc[[0, 5]] == [1.0, 2.0]))

    # with no contrast stimuli the c50 metric should be null
    assert('c50_dg' in dg.metrics.columns)
    assert(np.allclose(dg.metrics['c50_dg'].values, [np.nan]*6, equal_nan=True))

    assert('f1_f0_dg' in dg.metrics.columns)
    assert(np.allclose(dg.metrics['f1_f0_dg'].loc[[0, 1, 2, 3, 4, 5]],
                       [0.001572, np.nan, 1.999778, np.nan, 1.560436, 1.999978], equal_nan=True, atol=1.0e-06))

    assert('mod_idx_dg' in dg.metrics.columns)
    assert('g_osi_dg' in dg.metrics.columns)
    assert(np.allclose(dg.metrics['g_osi_dg'].loc[[0, 3, 4, 5]], [1.0, np.nan, 0.745356, 1.0], equal_nan=True))

    assert('g_dsi_dg' in dg.metrics.columns)
    assert(np.allclose(dg.metrics['g_dsi_dg'].loc[[0, 3, 4, 5]], [1.0, np.nan, 0.491209, 1.0], equal_nan=True))

    assert('firing_rate_dg' in dg.metrics.columns)
    assert('fano_dg' in dg.metrics.columns)
    assert('lifetime_sparseness_dg' in dg.metrics.columns)
    assert('run_pval_dg' in dg.metrics.columns)
    assert('run_mod_dg' in dg.metrics.columns)


def test_contrast_stimulus(ecephys_api_w_contrast):
    session = EcephysSession(api=ecephys_api_w_contrast)
    dg = DriftingGratings(ecephys_session=session)
    assert(len(dg.stim_table) == 40)

    assert(len(dg.stim_table_contrast) == 36)
    assert(len(dg.stimulus_conditions_contrast) == 36)
    assert(len(dg.conditionwise_statistics_contrast) == 36*6)


def test_metric_with_contrast(ecephys_api_w_contrast):
    session = EcephysSession(api=ecephys_api_w_contrast)
    dg = DriftingGratings(ecephys_session=session)

    assert(isinstance(dg.metrics, pd.DataFrame))
    assert(len(dg.metrics) == 6)
    assert(dg.metrics.index.names == ['unit_id'])

    # make sure normal prefered conditions remain the same
    assert('pref_ori_dg' in dg.metrics.columns)
    assert(np.all(dg.metrics['pref_ori_dg'].loc[[0, 1, 2, 3, 4, 5]] == np.full(6, 0.0)))
    assert('pref_tf_dg' in dg.metrics.columns)
    assert(np.all(dg.metrics['pref_tf_dg'].loc[[0, 5]] == [1.0, 2.0]))

    # Make sure class can see drifting_gratings_contrasts stimuli
    assert('c50_dg' in dg.metrics.columns)
    assert(np.allclose(dg.metrics['c50_dg'].loc[[0, 4, 5]], [0.359831, np.nan, 0.175859], equal_nan=True))


@pytest.mark.parametrize('response,tf,sampling_rate,expected',
                         [
                             (np.array([]), 2.0, 1000.0, np.nan),  # invalid input
                             (np.zeros(2000), 2.0, 1000.0, 0.0),  # no responses, MI ~ 0
                             (np.ones(2000), 4.0, 1000.0, 0.0),  # no derivation, MI ~ 0
                             (np.linspace(0.5, 12.1), 8.0, 1.0, np.nan),  # tf is outside niquist freq.
                             (np.array([0.1, 0.2, 0.2, 1.1]), 2.0, 4.0, 0.1389328986),  # low mi
                             (np.linspace(0.5, 12.1, 50), 8.0, 1000.0, 4.993941),  # high mi
                         ])
def test_modulation_index(response, tf, sampling_rate, expected):
    mi = modulation_index(response, tf, sampling_rate)  # return nan, invalid
    assert(np.isclose(mi, expected, equal_nan=True))


@pytest.mark.parametrize('contrast_vals,responses,expected',
                         [
                             (np.array([0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]), np.array([]), np.nan),  # invalid input
                             (np.array([0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]), np.full(9, 12.0), 0.0090),  # flat non-zero curve
                             (np.array([0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]), np.zeros(9), 0.3598313725490197),  # no responses
                             (np.array([0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]), np.linspace(0.0, 12.0, 9), 0.1330745098039216),
                             (np.array([0.01, 0.02, 0.04, 0.08, 0.13, 0.2, 0.35, 0.6, 1.0]), np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.nan),  # nan, special case where curve can't be fitted
                         ])
def test_c50(contrast_vals, responses, expected):
    c50_metric = c50(contrast_vals, responses)
    assert(np.isclose(c50_metric, expected, equal_nan=True))


@pytest.mark.parametrize('data_arr,tf,trial_duration,expected',
                         [
                             (np.array([]), 2.0, 1.0, np.nan),  # invalid input
                             (np.zeros((5, 256)), 4.0, 2.0, np.nan),  # no spikes
                             (np.ones((5, 256)), 18.0, 16.0, np.nan),  # tf*trial_duration is too high, returns nan
                             (np.full((5, 256), 5.0), 4.0, 2.0, 0.0),  # has constant spiking
                             (np.array([0, 0, 1, 1, 2, 0, 5, 1]), 2.0, 1.0, 0.894427190999916),  # can handle arrays
                             (np.array([[0, 0, 1, 1, 2, 0, 5, 1]]), 2.0, 1.0, 0.894427190999916)  # same as above but int matrix form
                         ])
def test_f1_f0(data_arr, tf, trial_duration, expected):
    f1_f0_val = f1_f0(data_arr, tf, trial_duration)
    assert(np.isclose(f1_f0_val, expected, equal_nan=True))


if __name__ == '__main__':
    # test_stimulus()
    test_metrics()
    # test_stim_table_contrast()
    # test_contrast_stimulus()
    # test_metric_with_contrast()

