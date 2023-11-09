import copy
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache.tables.util.prior_exposure_processing import (  # noqa: E501
    add_experience_level_ophys,
    get_prior_exposures_to_image_set,
    get_prior_exposures_to_omissions,
    get_prior_exposures_to_session_type,
)


def test_prior_exposure_to_session_type():
    """Tests normal behavior as well as case where session type is missing"""
    df = pd.DataFrame(
        {
            "session_type": ["A", "A", None, "A", "B"],
            "mouse_id": [0, 0, 0, 0, 1],
            "date_of_acquisition": [0, 1, 2, 3, 0],
        },
        index=pd.Series([0, 1, 2, 3, 4], name="behavior_session_id"),
    )
    expected = pd.Series(
        [0, 1, np.nan, 2, 0],
        index=pd.Series([0, 1, 2, 3, 4], name="behavior_session_id"),
    ).astype("Int64")
    obtained = get_prior_exposures_to_session_type(df=df)
    pd.testing.assert_series_equal(expected, obtained)


def test_prior_exposure_to_image_set():
    """Tests normal behavior as well as case where session type is not an
    image set type"""
    df = pd.DataFrame(
        {
            "session_type": [
                "TRAINING_1_images_A",
                "OPHYS_2_images_A_passive",
                "foo",
                "OPHYS_3_images_A",
                "B",
            ],
            "mouse_id": [0, 0, 0, 0, 1],
            "date_of_acquisition": [0, 1, 2, 3, 0],
        },
        index=pd.Index([0, 1, 2, 3, 4], name="behavior_session_id"),
    )
    expected = pd.Series(
        [0, 1, np.nan, 2, np.nan],
        index=pd.Series([0, 1, 2, 3, 4], name="behavior_session_id"),
    ).astype("Int64")
    obtained = get_prior_exposures_to_image_set(df=df)
    pd.testing.assert_series_equal(expected, obtained)


def test_prior_exposure_to_omissions():
    """Tests normal behavior and tests case where flash_omit_probability
    needs to be looked up for habituation session. Only 1 of the habituation
    sessions has omissions"""
    df = pd.DataFrame(
        {
            "session_type": [
                "OPHYS_1_images_A",
                "OPHYS_2_images_A_passive",
                "OPHYS_1_habituation",
                "OPHYS_2_habituation",
                "OPHYS_3_habituation",
            ],
            "mouse_id": [0, 0, 1, 1, 1],
            "foraging_id": [1, 2, 3, 4, 5],
            "date_of_acquisition": [0, 1, 0, 1, 2],
        },
        index=pd.Index([0, 1, 2, 3, 4], name="behavior_session_id"),
    )
    expected = pd.Series(
        [0, 1, 0, 0, 1],
        index=pd.Index([0, 1, 2, 3, 4], name="behavior_session_id"),
    ).astype("Int64")

    class MockFetchApi:
        def get_behavior_stage_parameters(self, foraging_ids):
            return {3: {}, 4: {"flash_omit_probability": 0.05}, 5: {}}

    fetch_api = MockFetchApi()
    obtained = get_prior_exposures_to_omissions(df=df, fetch_api=fetch_api)
    pd.testing.assert_series_equal(expected, obtained)


def test_add_experience_level():

    input_data = []
    expected_data = []

    datum = {'id': 0,
             'session_number': 1,
             'prior_exposures_to_image_set': 4,
             'session_type': 'OPHYS_1'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Familiar'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 1,
             'session_number': 2,
             'prior_exposures_to_image_set': 5,
             'session_type': 'OPHYS_2'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Familiar'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 2,
             'session_number': 3,
             'prior_exposures_to_image_set': 1772,
             'session_type': 'OPHYS_3'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Familiar'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 3,
             'session_number': 4,
             'prior_exposures_to_image_set': 0,
             'session_type': 'OPHYS_4'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel 1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 4,
             'session_number': 5,
             'prior_exposures_to_image_set': 0,
             'session_type': 'OPHYS_5'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel 1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 5,
             'session_number': 6,
             'prior_exposures_to_image_set': 0,
             'session_type': 'OPHYS_6'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel 1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 7,
             'session_number': 4,
             'prior_exposures_to_image_set': 2,
             'session_type': 'OPHYS_4'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel >1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 8,
             'session_number': 5,
             'prior_exposures_to_image_set': 1,
             'session_type': 'OPHYS_5'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel >1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 9,
             'session_number': 6,
             'prior_exposures_to_image_set': 3,
             'session_type': 'OPHYS_6'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'Novel >1'
    expected_data.append(copy.deepcopy(datum))

    datum = {'id': 10,
             'session_number': 7,
             'prior_exposures_to_image_set': 3,
             'session_type': 'OPHYS_7'}
    input_data.append(copy.deepcopy(datum))
    datum['experience_level'] = 'None'
    expected_data.append(copy.deepcopy(datum))

    input_df = pd.DataFrame(input_data)
    expected_df = pd.DataFrame(expected_data)
    output_df = add_experience_level_ophys(input_df)
    assert not input_df.equals(output_df)
    assert len(input_df.columns) != len(output_df.columns)
    assert output_df.equals(expected_df)
