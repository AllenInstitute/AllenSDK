import os
import numpy as np
import pandas as pd
import pytest
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc

cache_test_base = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/test_data'
cache_paths = {
    'manifest_path': os.path.join(cache_test_base, 'visual_behavior_data_manifest.csv'),
    'nwb_base_dir': os.path.join(cache_test_base, 'nwb_files'),
    'analysis_files_base_dir': os.path.join(cache_test_base, 'analysis_files'),
    'analysis_files_metadata_path': os.path.join(cache_test_base, 'analysis_files_metadata.json')
}
cache = bpc.BehaviorProjectCache(cache_paths)
session = cache.get_session(792815735)

# Test trials extra columns
def test_extra_trials_columns():
    for new_key in ['reward_rate', 'response_binary']:
        assert new_key in session.trials.keys()

def test_extra_stimulus_presentation_columns():
    for new_key in [
            'absolute_flash_number',
            'time_from_last_lick',
            'time_from_last_reward',
            'time_from_last_change',
            'block_index',
            'image_block_repetition',
            'repeat_within_block']:
        assert new_key in session.stimulus_presentations.keys()

def test_stimulus_presentations_image_set():
    # We made the image set just 'A' or 'B'
    assert session.stimulus_presentations['image_set'].unique() == np.array(['A'])

def test_stimulus_templates():
    # Was a dict with only one key, so we popped it out 
    assert isinstance(session.stimulus_templates, np.ndarray)

# Test trial response df
@pytest.mark.parametrize('key, output', [
    ('mean_response', 0.0053334),
    ('baseline_response', -0.0020357),
    ('p_value', 0.6478659),
])
def test_session_trial_response(key, output):
    trial_response = session.trial_response_df
    np.testing.assert_almost_equal(trial_response.loc[817103993].iloc[0][key], output, decimal=6)

@pytest.mark.parametrize('key, output', [
    ('time_from_last_lick', 7.3577),
    ('running_speed', 22.143871),
    ('duration', 0.25024),
])
def test_session_flash_response(key, output):
    flash_response = session.flash_response_df
    np.testing.assert_almost_equal(flash_response.loc[817103993].iloc[0][key], output, decimal=6)

def test_analysis_files_metadata():
    assert cache.analysis_files_metadata[
        'trial_response_df_params'
    ]['response_window_duration_seconds'] == 0.5

def test_session_image_loading():
    assert isinstance(session.max_projection.data, np.ndarray)

def test_no_invalid_rois():
    # We made the cache return sessions without the invalid rois
    assert session.cell_specimen_table['valid_roi'].all()

def test_get_container_sessions():
    container_id = cache.manifest['container_id'].unique()[0]
    container_sessions = cache.get_container_sessions(container_id)
    session = container_sessions['OPHYS_1_images_A']
    assert isinstance(session, bpc.ExtendedBehaviorSession)
    np.testing.assert_almost_equal(session.dff_traces.loc[817103993]['dff'][0], 0.3538657529565)

def test_cache_from_json():
    json_path = os.path.join(cache_test_base, 'behavior_ophys_cache.json')
    cache = bpc.BehaviorProjectCache.from_json(json_path)
    assert isinstance(cache, bpc.BehaviorProjectCache)
    assert isinstance(cache.manifest, pd.DataFrame)
