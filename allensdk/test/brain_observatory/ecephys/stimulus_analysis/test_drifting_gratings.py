import os
import numpy as np
import pandas as pd
import pytest
import itertools

from conftest import MockSessionApi
#from allensdk.brain_observatory.ecephys.stimulus_analysis import drifting_gratings
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.drifting_gratings import DriftingGratings

class MockDGSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        features = np.array(np.meshgrid([1.0, 2.0, 4.0, 8.0, 15.0],                            # TF
                                        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])  # ORI
                            ).reshape(2, 40)
                                        #[0.0, 0.25, 0.50, 0.75]))#.reshape(3, 120)  # Phase
        print(features)
        print(features.shape)
        exit()

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

def mock_ecephys_api():
    return MockDGSessionApi()


session = EcephysSession(api=mock_ecephys_api())
sg = DriftingGratings(ecephys_session=session)
print(sg.stim_table)

"""
data_dir = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh'

@pytest.mark.parametrize('spikes_nwb,expected_csv,analysis_params,units_filter',
                         [
                             #(os.path.join(data_dir, 'data', 'mouse406807_integration_test.spikes.nwb2'),
                             # os.path.join(data_dir, 'expected', 'mouse406807_integration_test.drifting_gratings.csv'),
                             # {'col_ori': 'ori'})
                             (os.path.join(data_dir, 'data', 'ecephys_session_773418906.nwb'),
                              os.path.join(data_dir, 'expected', 'ecephys_session_773418906.drifting_gratings.csv'),
                              {},
                              [914580284, 914580302, 914580328, 914580366, 914580360, 914580362, 914580380, 914580370,
                               914580408, 914580384, 914580402, 914580400, 914580396, 914580394, 914580392, 914580390,
                               914580382, 914580412, 914580424, 914580422, 914580438, 914580420, 914580434, 914580432,
                               914580428, 914580452, 914580450, 914580474, 914580470, 914580490])
                         ])
def test_metrics(spikes_nwb, expected_csv, analysis_params, units_filter, skip_cols=[]):
    ## Full intergration tests of metrics table
    # TODO: Test is only temporary while the stimulus_analysis modules is in development. Replace with unit tests and/or move to integration testing framework
    if not os.path.exists(spikes_nwb):
        pytest.skip('No input spikes file {}.'.format(spikes_nwb))
    if not os.access(spikes_nwb, os.R_OK):
        pytest.skip(f"can't access file at {spikes_nwb}")
    if not os.access(expected_csv, os.R_OK):
        pytest.skip(f"can't access file at {expected_csv}")

    np.random.seed(0)

    analysis_params = analysis_params or {}
    analysis = drifting_gratings.DriftingGratings(spikes_nwb, filter=units_filter, **analysis_params)
    # Make sure some of the non-metrics structures are returning valid(ish) tables
    assert(len(analysis.stim_table) > 1)
    assert(set(analysis.unit_ids) == set(units_filter))
    assert(len(analysis.running_speed) == len(analysis.stim_table))
    assert(analysis.stim_table_spontaneous.shape == (3, 5))
    assert(set(analysis.spikes.keys()) == set(units_filter))
    assert(len(analysis.conditionwise_psth) > 1)

    actual_data = analysis.metrics.sort_index()

    expected_data = pd.read_csv(expected_csv)
    expected_data = expected_data.set_index('unit_id')
    expected_data = expected_data.sort_index()  # in theory this should be sorted in the csv, no point in risking it.

    assert(np.all(actual_data.index.values == expected_data.index.values))
    assert(set(actual_data.columns) == (set(expected_data.columns) - set(skip_cols)))

    assert(np.allclose(actual_data['pref_ori_dg'].astype(np.float), expected_data['pref_ori_dg'], equal_nan=True))
    assert(np.allclose(actual_data['pref_tf_dg'].astype(np.float), expected_data['pref_tf_dg'], equal_nan=True))
    assert(np.allclose(actual_data['c50_dg'].astype(np.float), expected_data['c50_dg'], equal_nan=True))
    assert(np.allclose(actual_data['f1_f0_dg'].astype(np.float), expected_data['f1_f0_dg'], equal_nan=True))
    assert(np.allclose(actual_data['mod_idx_dg'].astype(np.float), expected_data['mod_idx_dg'], equal_nan=True))
    assert(np.allclose(actual_data['g_osi_dg'].astype(np.float), expected_data['g_osi_dg'], equal_nan=True))
    assert(np.allclose(actual_data['firing_rate_dg'].astype(np.float), expected_data['firing_rate_dg'], equal_nan=True))
    assert(np.allclose(actual_data['reliability_dg'].astype(np.float), expected_data['reliability_dg'], equal_nan=True))
    assert(np.allclose(actual_data['fano_dg'].astype(np.float), expected_data['fano_dg'], equal_nan=True))
    assert(np.allclose(actual_data['lifetime_sparseness_dg'].astype(np.float), expected_data['lifetime_sparseness_dg'],
                       equal_nan=True))
    assert(np.allclose(actual_data['run_pval_dg'].astype(np.float), expected_data['run_pval_dg'], equal_nan=True))

    assert(np.allclose(actual_data['run_mod_dg'].astype(np.float), expected_data['run_mod_dg'], equal_nan=True))


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
    orivals = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    tfvals = [1.0, 2.0, 4.0, 8.0, 15.0]
    fmatrix = np.zeros((40*2, 2))
    fmatrix[0:40, :] = np.array(list(itertools.product(orivals, tfvals)))
    return pd.DataFrame({'Ori': fmatrix[:, 0], 'TF': fmatrix[:, 1],
                         'stimulus_name': ['drifting_gratings']*40 + ['spontaneous']*40,
                         'start_time': np.linspace(5000.0, 5060.0, 40*2),
                         'stop_time': np.linspace(5000.0, 5060.0, 40*2) + 0.25,
                         'duration': 0.25})


@pytest.mark.skip()
def test_static_gratings(ecephys_session, stimulus_table):
    ecephys_session.stimulus_presentations = stimulus_table  # patch.object won't work since stimulus_presentations is a constructor variable.
    dg_obj = drifting_gratings.DriftingGratings(ecephys_session)
    assert(isinstance(dg_obj.stim_table, pd.DataFrame))
    assert(len(dg_obj.stim_table) == 40)
    assert(dg_obj.number_tf == 5)
    assert(np.all(dg_obj.tfvals == [1.0, 2.0, 4.0, 8.0, 15.0]))
    assert(dg_obj.number_ori == 8)
    assert (np.all(dg_obj.orivals == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]))
    assert(dg_obj.numbercells == 20)
    assert(dg_obj.mean_sweep_events.shape == (40, 20))


@pytest.mark.skip()
@pytest.mark.parametrize('response,trials,bias,expected',
                         [
                             (np.array([1.233, 2.4, 0.5667, 0.8, 0.5]), np.array(
                                 [1.5, 2.5, 1.0, 0.5, 0.0, 0.5, 2.0, 4.5, 2.5, 0.5, 1.5, 1.0, 0.5, 1.5, 0.0, 0.5]),
                              5, 0.4104327122),
                             (np.array([0.1, 0.233, 0.0333, 0.0667, 0.0333]),
                              np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5,
                              1.0)
                         ])
def test_get_tfdi(response, trials, bias, expected):
    assert (np.isclose(drifting_gratings.get_tfdi(response, trials, bias), expected))


@pytest.mark.skip()
@pytest.mark.parametrize('peak_response,all_responses,blank_responses,expected',
                         [
                             (0.2334, np.array([[0.0, 0.0, 0.0333, 0.0333, 0.0333],
                                                [0.033, 0.1, 0.033, 0.0333, 0.0333],
                                                [0.067, 0.0, 0.067, 0.0667, 0.0333],
                                                [0.033, 0.1, 0.067, 0.0, 0.0],
                                                [0.167, 0.033, 0.1, 0.133, 0.033],
                                                [0.0, 0.033, 0.0, 0.0, 0.1],
                                                [0.1, 0.233, 0.033, 0.0667, 0.033],
                                                [0.0, 0.0667, 0.067, 0.0667, 0.0]]), 0.03333333333333333,
                              (0.20006666666666667, 0.01745666666666667))
                         ])
def test_get_suppressed_contrast(peak_response, all_responses, blank_responses, expected):
    assert (np.allclose(drifting_gratings.get_suppressed_contrast(peak_response, all_responses, blank_responses), expected))


@pytest.mark.skip()
@pytest.mark.parametrize('tuning_responses,tf_values,pref_tf_index,expected',
                         [
                             (np.array([0.46667, 0.16667, 1.7, 0.9333, 0.4]), [1.0, 2.0, 4.0, 8.0, 15.0], 2,
                              (2.282457502110048, 4.865059677849396, 2.8284271247461903, np.nan)),
                             (np.array([17.633333, 17.96666667, 10.73333333, 12.63333333, 16.2]),
                              [1.0, 2.0, 4.0, 8.0, 15.0], 1, (np.nan, np.nan, np.nan, np.nan)),
                             (np.array([3.26666667, 2.7, 2.53333333, 0.83333333, 1.3]), [1.0, 2.0, 4.0, 8.0, 15.0],
                              0, (0, 1.0, np.nan, 6.498019170849885))
                         ])
def test_fit_tf_tuning(tuning_responses, tf_values, pref_tf_index, expected):
    assert (np.allclose(drifting_gratings.get_fit_tf_tuning(tuning_responses, tf_values, pref_tf_index), expected, equal_nan=True))
"""
