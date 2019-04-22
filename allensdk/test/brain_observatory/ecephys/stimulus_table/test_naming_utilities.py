import pytest
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.stimulus_table import naming_utilities as nu


@pytest.mark.parametrize(
    "table,expected",
    [
        [
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_four_more_repeats",
                        "natural_movie_four",
                        "natural_movie_shuffled",
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_four_more_repeats",
                        "natural_movie_four",
                        "natural_movie_four_shuffled",
                    ]
                }
            ),
        ],
        [
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_four_more_repeats",
                        "natural_movie_four",
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_four_more_repeats",
                        "natural_movie_four",
                    ]
                }
            ),
        ],
        [
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_4_more_repeats",
                        "natural_movie_4",
                        "natural_movie_shuffled",
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_4_more_repeats",
                        "natural_movie_4",
                        "natural_movie_4_shuffled",
                    ]
                }
            ),
        ],
    ],
)
def test_add_number_to_shuffled_movie(table, expected):
    obtained = nu.add_number_to_shuffled_movie(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize(
    "table,expected",
    [
        [
            pd.DataFrame(
                {"stimulus_name": ["natural_movie_4", "natural_movie_5_more_repeats"]}
            ),
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_movie_four",
                        "natural_movie_five_more_repeats",
                    ]
                }
            ),
        ]
    ],
)
def test_standardize_movie_numbers(table, expected):
    obtained = nu.standardize_movie_numbers(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize(
    "table,name_map,expected",
    [
        [
            pd.DataFrame({"stimulus_name": ["Natural Images", "contrast_response"]}),
            {
                "Natural Images": "natural_scenes",
                "contrast_response": "drifting_gratings_contrast",
            },
            pd.DataFrame(
                {"stimulus_name": ["natural_scenes", "drifting_gratings_contrast"]}
            ),
        ],
        [
            pd.DataFrame(
                {"stimulus_name": ["Natural Images", "contrast_response", np.nan]}
            ),
            {
                "Natural Images": "natural_scenes",
                "contrast_response": "drifting_gratings_contrast",
                None: "spontaneous",
            },
            pd.DataFrame(
                {
                    "stimulus_name": [
                        "natural_scenes",
                        "drifting_gratings_contrast",
                        "spontaneous",
                    ]
                }
            ),
        ],
    ],
)
def test_map_stimulus_names(table, name_map, expected):
    obtained = nu.map_stimulus_names(table, name_map)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize(
    "table,expected",
    [
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]}),
            pd.DataFrame({"a": [1, 2, 3]}),
        ],
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]}),
            pd.DataFrame({"a": [1, 2, 3]}),
        ],
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [None, None, 4]}),
            pd.DataFrame({"a": [1, 2, 3], "b": [None, None, 4]}),
        ],
    ],
)
def test_drop_empty_columns(table, expected):
    obtained = nu.drop_empty_columns(table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)


@pytest.mark.parametrize(
    "table,expected",
    [
        [
            pd.DataFrame({"a": [1, 2, np.nan], "A": [np.nan, None, 3]}),
            pd.DataFrame({"a": [1, 2, 3]}),
        ],
        [
            pd.DataFrame({"bar": [1, 2, np.nan], "Bar": [np.nan, None, 3]}),
            pd.DataFrame({"bar": [1, 2, 3]}),
        ],
        [
            pd.DataFrame({"bar": [1, 2, np.nan], "Bar": [np.nan, 4, 3]}),
            pd.DataFrame({"bar": [1, 2, np.nan], "Bar": [np.nan, 4, 3]}),
        ],
        [
            pd.DataFrame(
                {
                    "bar": [1, 2, np.nan],
                    "Bar": [np.nan, 4, 3],
                    "BAR": [np.nan, np.nan, 3],
                }
            ),
            pd.DataFrame({"bar": [1, 2, 3], "Bar": [np.nan, 4, 3]}),
        ],
    ],
)
def test_collapse_colimns(table, expected):
    obtained = nu.collapse_columns(table)
    pd.testing.assert_frame_equal(
        expected, obtained, check_like=True, check_dtype=False
    )
