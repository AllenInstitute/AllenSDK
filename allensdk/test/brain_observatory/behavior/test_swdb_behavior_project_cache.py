import os
import numpy as np
import pandas as pd
import pytest
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc


@pytest.fixture
def cache_test_base():
    return '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/test_data'

@pytest.fixture
def cache(cache_test_base):
    return bpc.BehaviorProjectCache(cache_test_base)

@pytest.fixture
def session(cache):
    return cache.get_session(792815735)

# Test trials extra columns
@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_extra_trials_columns(session):
    for new_key in ['reward_rate', 'response_binary']:
        assert new_key in session.trials.keys()


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_extra_stimulus_presentation_columns(session):
    for new_key in [
            'absolute_flash_number',
            'time_from_last_lick',
            'time_from_last_reward',
            'time_from_last_change',
            'block_index',
            'image_block_repetition',
            'repeat_within_block']:
        assert new_key in session.stimulus_presentations.keys()


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_stimulus_presentations_image_set(session):
    # We made the image set just 'A' or 'B'
    assert session.stimulus_presentations['image_set'].unique() == np.array(['A'])


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_stimulus_templates(session):
    # Was a dict with only one key, where the value was a 3d array.
    # We made it a dict with image names as keys and 2d arrs (the images) as values
    for image_name, image_arr in session.stimulus_templates.items():
        assert image_arr.ndim == 2

# Test trial response df
@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
@pytest.mark.parametrize('key, output', [
    ('mean_response', 0.0053334),
    ('baseline_response', -0.0020357),
    ('p_value', 0.6478659),
])
def test_session_trial_response(key, output, session):
    trial_response = session.trial_response_df
    np.testing.assert_almost_equal(trial_response.query("cell_specimen_id == 817103993").iloc[0][key], output, decimal=6)


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
@pytest.mark.parametrize('key, output', [
    ('time_from_last_lick', 7.3577),
    ('mean_running_speed', 22.143871),
    ('duration', 0.25024),
])
def test_session_flash_response(key, output, session):
    flash_response = session.flash_response_df
    np.testing.assert_almost_equal(flash_response.query("cell_specimen_id == 817103993").iloc[0][key], output, decimal=6)


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_analysis_files_metadata(cache):
    assert cache.analysis_files_metadata[
        'trial_response_df_params'
    ]['response_window_duration_seconds'] == 0.5


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_session_image_loading(session):
    assert isinstance(session.max_projection.data, np.ndarray)


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_no_invalid_rois(session):
    # We made the cache return sessions without the invalid rois
    assert session.cell_specimen_table['valid_roi'].all()


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_get_container_sessions(cache):
    container_id = cache.experiment_table['container_id'].unique()[0]
    container_sessions = cache.get_container_sessions(container_id)
    session = container_sessions['OPHYS_1_images_A']
    assert isinstance(session, bpc.ExtendedBehaviorSession)
    np.testing.assert_almost_equal(session.dff_traces.loc[817103993]['dff'][0], 0.3538657529565)


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_binarized_segmentation_mask_image(session):
    np.testing.assert_array_equal(
        np.unique(np.array(session.segmentation_mask_image.data).ravel()),
        np.array([0, 1])

    )


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_no_nan_flash_running_speed(session):
    assert not pd.isnull(session.stimulus_presentations['mean_running_speed']).any()


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_licks_correct_colname(session):
    assert session.licks.columns == ['timestamps']


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_rewards_correct_colname(session):
    assert (session.rewards.columns == ['timestamps', 'volume', 'autorewarded']).all()


@pytest.mark.skip(reason="deprecated")
@pytest.mark.requires_bamboo
def test_dff_traces_correct_colname(session):
    # This is a Friday-harbor specific change
    assert 'cell_roi_id' not in session.dff_traces.columns
