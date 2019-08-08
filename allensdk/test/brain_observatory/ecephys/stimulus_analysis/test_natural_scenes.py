import os
import pytest
import numpy as np
import pandas as pd
from mock import MagicMock

from allensdk.brain_observatory.ecephys.stimulus_analysis import natural_scenes as ns
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


data_dir = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh'


@pytest.mark.parametrize('spikes_nwb,expected_csv,analysis_params,units_filter',
                         [
                             #(os.path.join(data_dir, 'data', 'mouse406807_integration_test.spikes.nwb2'),
                             # os.path.join(data_dir, 'expected', 'mouse406807_integration_test.natural_scenes.csv'),
                             # {})
                             (os.path.join(data_dir, 'data', 'ecephys_session_773418906.nwb'),
                              os.path.join(data_dir, 'expected', 'ecephys_session_773418906.natural_scenes.csv'),
                              {},
                              [914580630, 914580280, 914580278, 914580634, 914580610, 914580290, 914580288, 914580286,
                               914580284, 914580282, 914580294, 914580330, 914580304, 914580292, 914580300, 914580298,
                               914580308, 914580306, 914580302, 914580316, 914580314, 914580312, 914580310, 914580318,
                               914580324, 914580322, 914580320, 914580328, 914580326, 914580334])
                         ])
def test_metrics(spikes_nwb, expected_csv, analysis_params, units_filter, skip_cols=[]):
    """Full intergration tests of metrics table"""
    if not os.path.exists(spikes_nwb):
        pytest.skip('No input spikes file {}.'.format(spikes_nwb))

    analysis_params = analysis_params or {}
    analysis = ns.NaturalScenes(spikes_nwb, filter=units_filter, **analysis_params)
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

    assert(np.allclose(actual_data['pref_image_ns'].astype(np.float), expected_data['pref_image_ns'], equal_nan=True))
    assert(np.allclose(actual_data['image_selectivity_ns'].astype(np.float), expected_data['image_selectivity_ns'],
                       equal_nan=True))
    assert(np.allclose(actual_data['firing_rate_ns'].astype(np.float), expected_data['firing_rate_ns'], equal_nan=True))
    assert(np.allclose(actual_data['fano_ns'].astype(np.float), expected_data['fano_ns'], equal_nan=True))
    assert(np.allclose(actual_data['time_to_peak_ns'].astype(np.float), expected_data['time_to_peak_ns'],
                       equal_nan=True))
    assert(np.allclose(actual_data['reliability_ns'].astype(np.float), expected_data['reliability_ns'], equal_nan=True))
    assert(np.allclose(actual_data['lifetime_sparseness_ns'].astype(np.float), expected_data['lifetime_sparseness_ns'],
                       equal_nan=True))
    assert(np.allclose(actual_data['run_pval_ns'].astype(np.float), expected_data['run_pval_ns'], equal_nan=True))
    assert(np.allclose(actual_data['run_mod_ns'].astype(np.float), expected_data['run_mod_ns'], equal_nan=True))



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
    images = np.empty(119*3*2)
    images[0:(119*3)] = np.repeat(np.arange(-1, 118), 3)
    images[(119*3):] = np.nan
    return pd.DataFrame({'Image': images,
                         'stimulus_name': ['Natural images_5']*119*3 + ['spontaneous']*119*3,
                         'start_time': np.linspace(5000.0, 5060.0, 119*3*2),
                         'stop_time': np.linspace(5000.0, 5060.0, 119*3*2) + 0.25,
                         'duration': 0.25})

@pytest.mark.skip()
def test_static_gratings(ecephys_session, stimulus_table):
    ecephys_session.stimulus_presentations = stimulus_table  # patch.object won't work since stimulus_presentations is a constructor variable.
    ns_obj = ns.NaturalScenes(ecephys_session)
    assert(isinstance(ns_obj.stim_table, pd.DataFrame))
    assert(len(ns_obj.stim_table) == 119*3)
    assert(ns_obj.number_images == 119)
    assert(ns_obj.number_nonblank == 118)
    assert(ns_obj.numbercells == 20)
    assert(ns_obj.mean_sweep_events.shape == (119*3, 20))


"""
@pytest.mark.parametrize('responses,number_nonblank,expected',
                         [
                             (np.array([1.04, 2.8, 0.88, 2.08, 2.0, 0.48, 1.36, 2.72, 1.04, 0.96, 2.16, 0.56, 0.56, 0.96, 1.28, 1.52, 1.28, 1.36, 2.56, 1.76, 1.36, 1.04, 2.16, 1.2, 0.96, 0.88, 1.28, 2.88,
                                        1.28, 2.0, 1.6, 2.32, 1.84, 0.8, 1.04, 2.08, 2.56, 1.28, 1.04, 0.72, 1.6, 1.6, 1.44, 0.8, 1.52, 1.28, 2.4, 1.52, 1.04, 2.72, 0.96, 1.44, 1.04, 1.6, 2.24, 1.28, 1.2,
                                        1.44, 0.64, 1.36, 1.68, 0.72, 5.36, 1.12, 1.84, 0.88, 2.8, 1.84, 1.36, 1.04, 3.44, 1.28, 1.36, 2.16, 1.28, 1.36, 1.52, 0.88, 0.88, 1.12, 1.36, 0.8, 2.48, 1.28, 1.6,
                                        1.52, 1.04, 0.48, 0.8, 0.48, 2.48, 2.48, 1.2, 0.72, 1.28, 2.16, 1.68, 0.8, 1.28, 3.68, 1.6, 1.2, 1.92, 1.28, 1.6, 1.28, 3.12, 3.28, 2.0, 0.88, 1.92, 2.32, 0.88, 1.6,
                                        1.84, 1.12, 1.68, 1.36, 1.2]),
                              118, 0.559033898),
                             (np.zeros(119), 118, 1.0)
                         ])
"""
@pytest.mark.skip()
def test_get_image_selectivity(responses, number_nonblank, expected):
    assert(np.isclose(ns.get_image_selectivity(responses, number_nonblank), expected))