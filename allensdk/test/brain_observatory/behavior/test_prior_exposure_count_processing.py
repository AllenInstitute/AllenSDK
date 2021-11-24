import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables.util \
    .prior_exposure_processing import \
    get_prior_exposures_to_session_type, get_prior_exposures_to_image_set, \
    get_prior_exposures_to_omissions


def test_prior_exposure_to_session_type():
    """Tests normal behavior as well as case where session type is missing"""
    df = pd.DataFrame({
        'session_type': ['A', 'A', None, 'A', 'B'],
        'mouse_id': [0, 0, 0, 0, 1],
        'date_of_acquisition': [0, 1, 2, 3, 0]
    }, index=pd.Series([0, 1, 2, 3, 4], name='behavior_session_id'))
    expected = pd.Series([0, 1, np.nan, 2, 0],
                         index=pd.Series([0, 1, 2, 3, 4],
                                         name='behavior_session_id'))
    obtained = get_prior_exposures_to_session_type(df=df)
    pd.testing.assert_series_equal(expected, obtained)


def test_prior_exposure_to_image_set():
    """Tests normal behavior as well as case where session type is not an
    image set type"""
    df = pd.DataFrame({
        'session_type': ['TRAINING_1_images_A', 'OPHYS_2_images_A_passive',
                         'foo', 'OPHYS_3_images_A', 'B'],
        'mouse_id': [0, 0, 0, 0, 1],
        'date_of_acquisition': [0, 1, 2, 3, 0]
    }, index=pd.Index([0, 1, 2, 3, 4], name='behavior_session_id'))
    expected = pd.Series([0, 1, np.nan, 2, np.nan],
                         index=pd.Series([0, 1, 2, 3, 4],
                                         name='behavior_session_id'))
    obtained = get_prior_exposures_to_image_set(df=df)
    pd.testing.assert_series_equal(expected, obtained)


def test_prior_exposure_to_omissions():
    """Tests normal behavior and tests case where flash_omit_probability
    needs to be looked up for habituation session. Only 1 of the habituation
    sessions has omissions"""
    df = pd.DataFrame({
        'session_type': ['OPHYS_1_images_A', 'OPHYS_2_images_A_passive',
                         'OPHYS_1_habituation', 'OPHYS_2_habituation',
                         'OPHYS_3_habituation'],
        'mouse_id': [0, 0, 1, 1, 1],
        'foraging_id': [1, 2, 3, 4, 5],
        'date_of_acquisition': [0, 1, 0, 1, 2]
    }, index=pd.Index([0, 1, 2, 3, 4], name='behavior_session_id'))
    expected = pd.Series([0, 1, 0, 0, 1],
                         index=pd.Index([0, 1, 2, 3, 4],
                                        name='behavior_session_id'))

    class MockFetchApi:
        def get_behavior_stage_parameters(self, foraging_ids):
            return {
                3: {},
                4: {'flash_omit_probability': 0.05},
                5: {}
            }
    fetch_api = MockFetchApi()
    obtained = get_prior_exposures_to_omissions(df=df, fetch_api=fetch_api)
    pd.testing.assert_series_equal(expected, obtained)
