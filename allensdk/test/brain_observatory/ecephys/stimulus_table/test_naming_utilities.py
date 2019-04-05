import pytest
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.stimulus_table import naming_utilities as nu


@pytest.mark.parametrize('table,expected', [
    [
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four', 'natural_movie_shuffled']}),
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four', 'natural_movie_four_shuffled']})
    ],
    [
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four']}),
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four']})
    ],
    [
        pd.DataFrame({'stimulus_name': ['natural_movie_4_more_repeats', 'natural_movie_4', 'natural_movie_shuffled']}),
        pd.DataFrame({'stimulus_name': ['natural_movie_4_more_repeats', 'natural_movie_4', 'natural_movie_4_shuffled']})
    ],
])
def test_add_number_to_shuffled_movie(table, expected):
    obtained = nu.add_number_to_shuffled_movie(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize('table,expected', [
    [
        pd.DataFrame({'stimulus_name': ['natural_movie_4', 'natural_movie_5_more_repeats']}),
        pd.DataFrame({'stimulus_name': ['natural_movie_four', 'natural_movie_five_more_repeats']})
    ]
])
def test_standardize_movie_numbers(table, expected):
    obtained = nu.standardize_movie_numbers(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize('table,expected', [
    [
        pd.DataFrame({'stimulus_name': ['gabor_20_deg_250ms', 'gabor_53.1deg', 'gabor_.2_deg_241ms']}),
        pd.DataFrame({'stimulus_name': ['gabor', 'gabor', 'gabor'], 'diameter': [20.0, 53.1, 0.2]})
    ]
])
def test_extract_gabor_parameters(table, expected):
    obtained = nu.extract_gabor_parameters(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)



@pytest.mark.parametrize('table,name_map,expected', [
    [
        pd.DataFrame({'stimulus_name': ['Natural Images', 'contrast_response']}),
        {'Natural Images': 'natural_scenes', 'contrast_response': 'drifting_gratings_contrast'},
        pd.DataFrame({'stimulus_name': ['natural_scenes', 'drifting_gratings_contrast']}),
    ],
    [
        pd.DataFrame({'stimulus_name': ['Natural Images', 'contrast_response', np.nan]}),
        {'Natural Images': 'natural_scenes', 'contrast_response': 'drifting_gratings_contrast', None: 'spontaneous'},
        pd.DataFrame({'stimulus_name': ['natural_scenes', 'drifting_gratings_contrast', 'spontaneous']}),
    ]
])
def test_map_stimulus_names(table, name_map, expected):
    obtained = nu.map_stimulus_names(table, name_map)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)