import pytest
import os
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.stimulus_analysis import receptive_field_mapping as rfm


data_dir = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh'


@pytest.mark.parametrize('spikes_nwb,expected_csv,analysis_params,units_filter',
                         [
                             #(os.path.join(data_dir, 'data', 'mouse406807_integration_test.spikes.nwb2'),
                             # os.path.join(data_dir, 'expected', 'mouse406807_integration_test.receptive_field_mapping.csv'),
                             # {})
                             (os.path.join(data_dir, 'data', 'ecephys_session_773418906.nwb'),
                              os.path.join(data_dir, 'expected', 'ecephys_session_773418906.receptive_field_mapping.csv'),
                              {},
                              [914580284, 914580302, 914580328, 914580366, 914580360, 914580362, 914580380, 914580370,
                               914580408, 914580384, 914580402, 914580400, 914580396, 914580394, 914580392, 914580390,
                               914580382, 914580412, 914580424, 914580422, 914580438, 914580420, 914580434, 914580432,
                               914580428, 914580452, 914580450, 914580474, 914580470, 914580490])
                         ])
def test_metrics(spikes_nwb, expected_csv, analysis_params, units_filter, skip_cols=[]):
    """Full intergration tests of metrics table"""
    if not os.path.exists(spikes_nwb):
        pytest.skip('No input spikes file {}.'.format(spikes_nwb))
    if not os.access(spikes_nwb, os.R_OK):
        pytest.skip(f"can't access file at {spikes_nwb}")
    if not os.access(expected_csv, os.R_OK):
        pytest.skip(f"can't access file at {expected_csv}")

    np.random.seed(0)

    analysis_params = analysis_params or {}
    analysis = rfm.ReceptiveFieldMapping(spikes_nwb, filter=units_filter, **analysis_params)
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

    assert(np.allclose(actual_data['azimuth_rf'].astype(np.float), expected_data['azimuth_rf'], equal_nan=True))
    assert(np.allclose(actual_data['elevation_rf'].astype(np.float), expected_data['elevation_rf'], equal_nan=True))
    assert(np.allclose(actual_data['width_rf'].astype(np.float), expected_data['width_rf'], equal_nan=True))
    assert(np.allclose(actual_data['height_rf'].astype(np.float), expected_data['height_rf'], equal_nan=True))
    assert(np.allclose(actual_data['area_rf'].astype(np.float), expected_data['area_rf'], equal_nan=True))
    assert(np.allclose(actual_data['p_value_rf'].astype(np.float), expected_data['p_value_rf'], equal_nan=True))
    assert (np.all(actual_data['on_screen_rf'].astype(np.bool) == expected_data['on_screen_rf'].astype(np.bool)))
    assert(np.allclose(actual_data['firing_rate_rf'].astype(np.float), expected_data['firing_rate_rf'], equal_nan=True))
    assert(np.allclose(actual_data['fano_rf'].astype(np.float), expected_data['fano_rf'], equal_nan=True))
    assert(np.allclose(actual_data['time_to_peak_rf'].astype(np.float), expected_data['time_to_peak_rf'], equal_nan=True))
    assert(np.allclose(actual_data['reliability_rf'].astype(np.float), expected_data['reliability_rf'], equal_nan=True))
    assert(np.allclose(actual_data['lifetime_sparseness_rf'].astype(np.float), expected_data['lifetime_sparseness_rf'],
                       equal_nan=True))
    assert(np.allclose(actual_data['run_mod_rf'].astype(np.float), expected_data['run_mod_rf'], equal_nan=True))
    assert(np.allclose(actual_data['run_pval_rf'].astype(np.float), expected_data['run_pval_rf'], equal_nan=True))
