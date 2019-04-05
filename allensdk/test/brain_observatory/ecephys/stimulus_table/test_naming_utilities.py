import pytest
import pandas as pd

from allensdk.brain_observatory.ecephys.stimulus_table import naming_utilities as nu


@pytest.mark.parametrize('table,expected', [
    [
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four', 'natural_movie_shuffled']}),
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four', 'natural_movie_four_shuffled']})
    ],
    [
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four']}),
        pd.DataFrame({'stimulus_name': ['natural_movie_four_more_repeats', 'natural_movie_four']})
    ]
])
def test_add_number_to_shuffled_movie(table, expected):

    obtained = nu.add_number_to_shuffled_movie(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)