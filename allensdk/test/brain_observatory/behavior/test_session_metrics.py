import pytest
from allensdk.brain_observatory.behavior import session_metrics as metrics
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "trials, detect_col, trial_types, expected",
    [
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True]}),
            "detect",
            ["go", "catch"],
            0.75,
        ),
        (
            pd.DataFrame({"trial_type":[ "go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True]}),
            "detect",
            ["go"],
            0.5,
        ),
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True]}),
            "detect",
            [],
            0.8,
        ),
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],
                          "detect": [True, False, True, True, True]}),
            "detect",
            ["early"],
            np.nan,
        ),
    ],
)
def test_response_bias(trials, detect_col, trial_types, expected):
    assert metrics.response_bias(trials, detect_col, trial_types) == \
        pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize(
    "trials, expected",
    [
        (
            pd.DataFrame({"trial_type": ["go", "go", "catch", "catch", "aborted"],}),
            4,
        ),
        (
            pd.DataFrame({"trial_type":[ "go", "go", "go"],}),
            3,
        ),
        (
            pd.DataFrame({"trial_type": ["catch"],}),
            1,
        ),
        (
            pd.DataFrame({"trial_type": [],}),
            0,
        ),
        (
            pd.DataFrame({"trial_type": ["aborted", "nogo"]}),
            0,
        )
    ],
)
def test_num_contingent_trials(trials, expected):
    assert metrics.num_contingent_trials(trials) == expected
