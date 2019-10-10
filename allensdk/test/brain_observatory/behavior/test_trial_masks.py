import pytest
from allensdk.brain_observatory.behavior import trial_masks as masks
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "trials, trial_types, expected",
    [
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True],}),
            ["go", "catch"],
            pd.Series([True, True, True, True, False], name="trial_type"),
        ),
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True],}),
            ["aborted"],
            pd.Series([False, False, False, False, True], name="trial_type")
        ),
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True],}),
            [],
            pd.Series([True, True, True, True, True], name="trial_type"),
        ),
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True],}),
            ["early"],
            pd.Series([False, False, False, False, False], name="trial_type"),
        ),
        (
            pd.DataFrame({"trial_type": [],
                          "detect": [],}),
            ["go", "catch"],
            pd.Series([], name="trial_type"),
        ),
    ],
)
def test_trial_types(trials, trial_types, expected):
    pd.testing.assert_series_equal(
        masks.trial_types(trials, trial_types), expected, check_dtype=False)



@pytest.mark.parametrize(
    "trials, trial_types, expected",
    [
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "include": [True, False, True, True, True],}),
            ["go", "catch"],
            pd.Series([True, True, True, False], name="trial_type",
            index=[0, 2, 3, 4]),
        ),
    ],
)
def test_trial_types_works_with_subselection(trials, trial_types, expected):
    pd.testing.assert_series_equal(
        masks.trial_types(trials[trials["include"]], trial_types), expected, 
        check_dtype=False)


@pytest.mark.parametrize(
    "trials, expected",
    [
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True],}),
            pd.Series([True, True, True, True, False], name="trial_type"),
        ),
        (
            pd.DataFrame({"trial_type": [],
                          "detect": [],}),
            pd.Series([], name="trial_type"),
        ),
    ]
)
def test_contingent_trials(trials, expected):
    pd.testing.assert_series_equal(
        masks.contingent_trials(trials), expected, check_dtype=False)


@pytest.mark.parametrize(
    "trials, thresh, expected",
    [
        (
            pd.DataFrame({"reward_rate": [0.0, 1.0, 2.0, 3.0]}),
            -1.0,
            pd.Series([True, True, True, True], name="reward_rate"),
        ),
        (
            pd.DataFrame({"reward_rate": [0.0, 1.0, 2.0, 3.0]}),
            1.0,
            pd.Series([False, False, True, True], name="reward_rate"),
        ),
        (
            pd.DataFrame({"reward_rate": [0.0, 1.0, 2.0, 3.0]}),
            3.0,
            pd.Series([False, False, False, False], name="reward_rate"),
        ),
    ]
)
def test_reward_rate(trials, thresh, expected):
    pd.testing.assert_series_equal(masks.reward_rate(trials, thresh), expected)