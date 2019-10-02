import pytest
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.stimulus_analysis.static_gratings import StaticGratings, get_sfdi, fit_sf_tuning


class MockSGSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        features = np.array(np.meshgrid([0.02, 0.04, 0.08, 0.16, 0.32],            # SF
                                        [0.0, 30.0, 60.0, 90.0, 120.0, 150.0],     # ORI
                                        [0.0, 0.25, 0.50, 0.75])).reshape(3, 120)  # Phase

        return pd.DataFrame({
            'start_time': np.concatenate(([0.0], np.linspace(0.5, 30.25, 120, endpoint=True), [31.5])),
            'stop_time': np.concatenate(([0.5], np.linspace(0.75, 30.50, 120, endpoint=True), [32.0])),
            'stimulus_name': ['spontaneous'] + ['static_gratings']*120 + ['spontaneous'],
            'stimulus_block': [0] + [1]*120 + [0],
            'duration': [0.5] + [0.25]*120 + [0.5],
            'stimulus_index': [0] + [1]*120 + [0],
            'spatial_frequency': np.concatenate(([np.nan], features[0, :], [np.nan])),
            'orientation': np.concatenate(([np.nan], features[1, :], [np.nan])),
            'phase': np.concatenate(([np.nan], features[2, :], [np.nan]))
        }, index=pd.Index(name='id', data=np.arange(122)))


@pytest.fixture
def ecephys_api():
    return MockSGSessionApi()


def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    sg = StaticGratings(ecephys_session=session)
    assert(sg.name == 'Static Gratings')
    assert(set(sg.unit_ids) == set(range(6)))
    assert(len(sg.conditionwise_statistics) == 120*6)
    assert(sg.conditionwise_psth.shape == (120, 249, 6))
    assert(not sg.presentationwise_spike_times.empty)
    assert(len(sg.presentationwise_statistics) == 120*6)
    assert(len(sg.stimulus_conditions) == 120)


def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    sg = StaticGratings(ecephys_session=session)
    assert(isinstance(sg.stim_table, pd.DataFrame))
    assert(len(sg.stim_table) == 120)
    assert(set(sg.stim_table.columns).issuperset({'spatial_frequency', 'orientation', 'phase', 'start_time', 'stop_time'}))

    assert(set(sg.sfvals) == {0.02, 0.04, 0.08, 0.16, 0.32})
    assert(sg.number_sf == 5)

    assert(set(sg.orivals) == {0.0, 30.0, 60.0, 90.0, 120.0, 150.0})
    assert(sg.number_ori == 6)

    assert(set(sg.phasevals) == {0.0, 0.25, 0.50, 0.75})
    assert(sg.number_phase == 4)


def test_bad_stimulus_key(ecephys_api):
    with pytest.raises(Exception):
        session = EcephysSession(api=ecephys_api)
        sg = StaticGratings(ecephys_session=session, stimulus_key='gratings static')
        sg.stim_table


def test_bad_col_key(ecephys_api):
    with pytest.raises(KeyError):
        session = EcephysSession(api=ecephys_api)
        sg = StaticGratings(ecephys_session=session, col_sf='spatial_frequency', col_phase='esahp')
        sg.phasevals


def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    sg = StaticGratings(ecephys_session=session)
    assert(isinstance(sg.metrics, pd.DataFrame))
    assert(len(sg.metrics) == 6)
    assert(sg.metrics.index.names == ['unit_id'])

    assert('pref_sf_sg' in sg.metrics.columns)
    assert(np.all(sg.metrics['pref_sf_sg'].loc[[0, 2, 4]] == [0.02, 0.02, 0.04]))

    assert('pref_ori_sg' in sg.metrics.columns)
    assert(np.all(sg.metrics['pref_ori_sg'].loc[[0, 2, 4]] == [0.0, 0.0, 0.0]))

    assert('pref_phase_sg' in sg.metrics.columns)
    assert(np.all(sg.metrics['pref_phase_sg'].loc[[0, 1, 2, 3]] == [0.25, 0.75, 0.5, 0.0]))

    assert('g_osi_sg' in sg.metrics.columns)
    assert('time_to_peak_sg' in sg.metrics.columns)
    assert('firing_rate_sg' in sg.metrics.columns)
    assert('fano_sg' in sg.metrics.columns)
    assert('lifetime_sparseness_sg' in sg.metrics.columns)
    assert('run_pval_sg' in sg.metrics.columns)
    assert('run_mod_sg' in sg.metrics.columns)


@pytest.mark.parametrize('sf_tuning_responses,mean_sweeps_trials,expected',
                         [
                             (np.array([18.08333, 19.8333, 28.333, 14.80, 9.6170]),
                              np.array([12.0, 4.0, 8.0, 32.0, 4.0, 0.0, 4.0, 8.0, 24.0, 40.0, 32.0, 8.0, 20.0, 28.0,
                                        24.0, 28.0, 0.0, 4.0, 4.0, 24.0, 16.0, 8.0, 16.0, 4.0, 0.0, 4.0, 24.0, 4.0,
                                        12.0, 20.0, 0.0, 12.0, 0.0, 16.0]), 0.4402349784724991)
                         ])
def test_get_sfdi(sf_tuning_responses, mean_sweeps_trials, expected):
    assert(get_sfdi(sf_tuning_responses, mean_sweeps_trials, len(sf_tuning_responses)) == expected)


@pytest.mark.parametrize('sf_tuning_response,sf_vals,pref_sf_index,expected',
                         [
                             (np.array([2.69565217, 3.91836735, 2.36734694, 1.52, 2.21276596]),
                              [0.02, 0.04, 0.08, 0.16, 0.32], 1, (0.22704947240176027, 0.0234087755414, np.nan, np.nan)
                              ),
                             (np.array([1.14285714, 0.73469388, 7.44, 13.6, 11.6]), [0.02, 0.04, 0.08, 0.16, 0.32], 3,
                              (3.290141840632274, 0.1956416782774323, 0.08, np.nan)),
                             (np.array([2.24, 1.83333333, 1.68, 1.87755102, 1.87755102]),
                              [0.02, 0.04, 0.08, 0.16, 0.32], 0, (0.0, 0.019999999552965164, np.nan, 0.32))
                         ])
def test_fit_sf_tuning(sf_tuning_response, sf_vals, pref_sf_index, expected):
    assert(np.allclose(fit_sf_tuning(sf_tuning_response, sf_vals, pref_sf_index), expected, equal_nan=True))


if __name__ == '__main__':
    # test_stimulus()
    # test_load()
    # test_bad_stimulus_key()
    # test_bad_col_key()
    test_metrics()
    pass

