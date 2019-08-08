import pytest
import os
import pandas as pd
import numpy as np
import itertools
from mock import MagicMock, patch

from allensdk.brain_observatory.ecephys.stimulus_analysis import static_gratings
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


pd.set_option('display.max_columns', None)
data_dir = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh'


@pytest.mark.parametrize('spikes_nwb,expected_csv,analysis_params,units_filter',
                         [
                             #(os.path.join(data_dir, 'data', 'mouse406807_integration_test.spikes.nwb2'),
                             # os.path.join(data_dir, 'expected', 'mouse406807_integration_test.static_gratings.csv'),
                             # {'col_ori': 'ori', 'col_sf': 'sf', 'col_phase': 'phase'},
                             # None)

                             (os.path.join(data_dir, 'data', 'ecephys_session_773418906.nwb'),
                              os.path.join(data_dir, 'expected', 'ecephys_session_773418906.static_gratings.csv'),
                              {},
                              [914580630, 914580280, 914580278, 914580634, 914580610, 914580290, 914580288, 914580286,
                               914580284, 914580282, 914580294, 914580330, 914580304, 914580292, 914580300, 914580298,
                               914580308, 914580306, 914580302, 914580316, 914580314, 914580312, 914580310, 914580318,
                               914580324, 914580322, 914580320, 914580328, 914580326, 914580334])
                         ])
def test_metrics(spikes_nwb, expected_csv, analysis_params, units_filter, skip_cols=[]):
    """Full intergration tests of metrics table"""
    # TODO: Test is only temporary while the stimulus_analysis modules is in development. Replace with unit tests and/or move to integration testing framework
    if not os.path.exists(spikes_nwb):
        pytest.skip('No input spikes file {}.'.format(spikes_nwb))

    np.random.seed(0)  # required by

    analysis_params = analysis_params or {}
    analysis = static_gratings.StaticGratings(spikes_nwb, filter=units_filter, **analysis_params)
    # Make sure some of the non-metrics structures are returning valid(ish) tables
    assert(len(analysis.stim_table) > 1)
    assert(set(analysis.unit_ids) == set(units_filter))
    assert(len(analysis.running_speed) == len(analysis.stim_table))
    assert(analysis.stim_table_spontaneous.shape == (3, 5))
    assert(set(analysis.spikes.keys()) == set(units_filter))
    assert(len(analysis.conditionwise_psth) > 1)

    # Test the metrics() table is returning consistant values
    actual_data = analysis.metrics.sort_index()

    expected_data = pd.read_csv(expected_csv)
    expected_data = expected_data.set_index('unit_id')
    expected_data = expected_data.sort_index()  # in theory this should be sorted in the csv, no point in risking it.

    assert (np.all(actual_data.index.values == expected_data.index.values))
    assert (set(actual_data.columns) == (set(expected_data.columns) - set(skip_cols)))

    assert(np.allclose(actual_data['pref_sf_sg'].astype(np.float), expected_data['pref_sf_sg'], equal_nan=True))
    assert(np.allclose(actual_data['pref_ori_sg'].astype(np.float), expected_data['pref_ori_sg'], equal_nan=True))
    assert(np.allclose(actual_data['pref_phase_sg'].astype(np.float), expected_data['pref_phase_sg'], equal_nan=True))
    assert(np.allclose(actual_data['g_osi_sg'].astype(np.float), expected_data['g_osi_sg'], equal_nan=True))
    assert(np.allclose(actual_data['pref_sf_sg'].astype(np.float), expected_data['pref_sf_sg'], equal_nan=True))
    assert(np.allclose(actual_data['time_to_peak_sg'].astype(np.float), expected_data['time_to_peak_sg'],
                       equal_nan=True))
    assert(np.allclose(actual_data['firing_rate_sg'].astype(np.float), expected_data['firing_rate_sg'], equal_nan=True))
    assert(np.allclose(actual_data['reliability_sg'].astype(np.float), expected_data['reliability_sg'], equal_nan=True))
    assert(np.allclose(actual_data['fano_sg'].astype(np.float), expected_data['fano_sg'], equal_nan=True))
    assert(np.allclose(actual_data['lifetime_sparseness_sg'].astype(np.float), expected_data['lifetime_sparseness_sg'],
                       equal_nan=True))
    assert(np.allclose(actual_data['run_pval_sg'].astype(np.float), expected_data['run_pval_sg'], equal_nan=True))
    assert(np.allclose(actual_data['run_mod_sg'].astype(np.float), expected_data['run_mod_sg'], equal_nan=True))


@pytest.fixture
def ecephys_session():
    ecephys_ses = MagicMock(spec=EcephysSession)
    units_df = pd.DataFrame({'unit_id': np.arange(20)})
    units_df = units_df.set_index('unit_id')
    ecephys_ses.units = units_df
    ecephys_ses.spike_times = {uid: np.linspace(0, 1.0, 5) for uid in np.arange(20)}
    return ecephys_ses


@pytest.fixture
def stimulus_table():
    orival = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    sfvals = [0.02, 0.04, 0.08, 0.16, 0.32]
    phasevals = [0.0, 0.25, 0.50, 0.75]
    fmatrix = np.zeros((120*2, 3))
    fmatrix[0:120, :] = np.array(list(itertools.product(orival, sfvals, phasevals)))
    return pd.DataFrame({'Ori': fmatrix[:, 0], 'SF': fmatrix[:, 1], 'Phase': fmatrix[:, 2],
                         'stimulus_name': ['static_gratings_6']*120 + ['spontaneous']*120,
                         'start_time': np.linspace(5000.0, 5060.0, 120*2),
                         'stop_time': np.linspace(5000.0, 5060.0, 120*2) + 0.25,
                         'duration': 0.25})

@pytest.mark.skip(reason='Turning off until class is completed.')
# @patch.object(EcephysSession, 'stimulus_presentations', stimulus_table())
def test_static_gratings(ecephys_session, stimulus_table):
    ecephys_session.stimulus_presentations = stimulus_table  # patch.object won't work since stimulus_presentations is a constructor variable.
    sg_obj = static_gratings.StaticGratings(ecephys_session)
    assert(isinstance(sg_obj.stim_table, pd.DataFrame))
    assert(len(sg_obj.stim_table) == 120)
    assert(sg_obj.number_sf == 5)
    assert(np.all(sg_obj.sfvals == [0.02, 0.04, 0.08, 0.16, 0.32]))
    assert(sg_obj.number_ori == 6)
    assert (np.all(sg_obj.orivals == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]))
    assert(sg_obj.number_phase == 4)
    assert(np.all(sg_obj.phasevals == [0.0, 0.25, 0.50, 0.75]))
    assert(sg_obj.numbercells == 20)
    assert(sg_obj.mean_sweep_events.shape == (120, 20))


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
    assert(np.allclose(static_gratings.fit_sf_tuning(sf_tuning_response, sf_vals, pref_sf_index), expected,
                       equal_nan=True))


@pytest.mark.parametrize('sf_tuning_responses,mean_sweeps_trials,expected',
                         [
                             (np.array([18.08333, 19.8333, 28.333, 14.80, 9.6170]),
                              np.array([12.0, 4.0, 8.0, 32.0, 4.0, 0.0, 4.0, 8.0, 24.0, 40.0, 32.0, 8.0, 20.0, 28.0,
                                        24.0, 28.0, 0.0, 4.0, 4.0, 24.0, 16.0, 8.0, 16.0, 4.0, 0.0, 4.0, 24.0, 4.0,
                                        12.0, 20.0, 0.0, 12.0, 0.0, 16.0]), 0.4402349784724991)
                         ])
def test_get_sfdi(sf_tuning_responses, mean_sweeps_trials, expected):
    assert(static_gratings.get_sfdi(sf_tuning_responses, mean_sweeps_trials, len(sf_tuning_responses)) == expected)


if __name__ == '__main__':
    test_metrics(os.path.join(data_dir, 'data', 'mouse406807_integration_test.spikes.nwb2'),
                 os.path.join(data_dir, 'expected', 'mouse406807_integration_test.static_gratings.csv'),
                 {'col_ori': 'ori', 'col_sf': 'sf', 'col_phase': 'phase'})
