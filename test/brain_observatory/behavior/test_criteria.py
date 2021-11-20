import pytest
import pandas as pd
from allensdk.brain_observatory.behavior import criteria
from allensdk.core.exceptions import DataFrameKeyError, DataFrameIndexError


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 0, 1: 0, 2: 0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 2.0, 1: 2.0, 2: 2.0, },
            }),
            False,
        ),  # should need to be greater than 2.0
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 0, 1: 0, 2: 2.1, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 0, 1: 2.1, 2: 2.1, },
            }),
            True,
        ),
    ],
)
def test_two_out_of_three_aint_bad(session_summary, expected):
    assert criteria.two_out_of_three_aint_bad(session_summary) == expected


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, },
                "dprime_peak": {0: 0, 1: 2.1, }
            }),
            pytest.raises(DataFrameIndexError),
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "other_col": {0: 0, 1: 2.1, 2: 2.1, }
            }),
            pytest.raises(DataFrameKeyError),
        ),
    ],
)
def test_two_out_of_three_aint_bad_exception(session_summary, expected):
    with expected:
        criteria.two_out_of_three_aint_bad(session_summary)


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 0, 1: 0, 2: 0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 2.0, 1: 2.0, 2: 2.0, },
            }),
            False,
        ),  # should need to be greater than 2.0
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 0, 1: 2.1, 2: 0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "dprime_peak": {0: 0, 1: 0, 2: 2.1, },
            }),
            True,
        ),

    ],
)
def test_yesterday_was_good(session_summary, expected):
    assert criteria.yesterday_was_good(session_summary) == expected


@pytest.mark.parametrize(
    "session_summary,expected",
    [
        (
            pd.DataFrame({
                "training_day": {},
                "dprime_peak": {},
            }),
            pytest.raises(DataFrameIndexError),
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "other_col": {0: 0, 1: 0, 2: 2.1, },
            }),
            pytest.raises(DataFrameKeyError),
        ),
    ],
)
def test_yesterday_was_good_exception(session_summary, expected):
    with expected:
        criteria.yesterday_was_good(session_summary)


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.1, },
            }),
            False,
        ),  # non-inclusive
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.9, },
            }),
            False,
        ),  # non-inclusive
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.11, },
            }),
            True,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.89, },
            }),
            True,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.1011, },
            }),
            True,
        ),  # doesn't round
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "response_bias": {0: 0.0, 1: 0.0, 2: 0.8999, },
            }),
            True,
        ),  # doesn't round
    ],
)
def test_no_response_bias(session_summary, expected):
    assert criteria.no_response_bias(session_summary) == expected


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {},
                "response_bias": {},
            }),
            pytest.raises(DataFrameIndexError),
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "other_col": {0: 0.0, 1: 0.0, 2: 0.11, },
            }),
            pytest.raises(DataFrameKeyError),
        ),
    ],
)
def test_no_response_bias_exception(session_summary, expected):
    with expected:
        criteria.no_response_bias(session_summary)


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "num_contingent_trials": {0: 0.0, 1: 0.0, 2: 0.0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "num_contingent_trials": {0: 0.0, 1: 0.0, 2: 100.0, },
            }),
            False,
        ),  # non-inclusive
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "num_contingent_trials": {0: 0.0, 1: 0.0, 2: 300.0, },
            }),
            False,
        ),  # non-inclusive
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "num_contingent_trials": {0: 0.0, 1: 0.0, 2: 301.0, },
            }),
            True,
        ),
    ],
)
def test_whole_lotta_trials(session_summary, expected):
    assert criteria.whole_lotta_trials(session_summary) == expected


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "training_day": {},
                "num_contingent_trials": {},
            }),
            pytest.raises(DataFrameIndexError),
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0, 1: 1, 2: 3, },
                "other_col": {0: 0.0, 1: 0.0, 2: 301.0, },
            }),
            pytest.raises(DataFrameKeyError),
        ),
    ]
)
def test_whole_lotta_trials_exception(session_summary, expected):
    with expected:
        criteria.whole_lotta_trials(session_summary)


@pytest.mark.parametrize(
    "trials, expected",
    [
        (
            pd.DataFrame({
                "training_day": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, },  # associate all with same training day
                "trial_type": {0: "aborted", 1: "go", 2: "catch", 3: "go", },
                "trial_length": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, },
            }),
            True,
        ),
        (
            pd.DataFrame({
                "training_day": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, },  # associate all with same training day
                "trial_type": {0: "aborted", 1: "go", 2: "catch", 3: "aborted", },
                "trial_length": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, },
            }),
            False,
        ),  # non-inclusive
        (
            pd.DataFrame({
                "training_day": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, },  # associate all with same training day
                "trial_type": {0: "aborted", 1: "go", 2: "aborted", 3: "aborted", },
                "trial_length": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "training_day": {},  # associate all with same training day
                "trial_type": {},
                "trial_length": {},
            }),
            False,
        ),
    ],
)
def test_mostly_useful(trials, expected):
    assert criteria.mostly_useful(trials) == expected


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "task": {0: "Images", 1: "", 2: "", 3: "Images", },
                "dprime_peak": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, },
                "num_engaged_trials": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "task": {0: "", 1: "", 2: "", 3: "Images", 4: "Images", 5: "Images"},
                "dprime_peak": {0: 1.2, 1: 2.0, 2: 1.1, 3: 1.1, 4: 2.0, 5: 1.2, },
                "num_engaged_trials": {0: 101, 1: 102, 2: 200, 3: 101, 4: 102, 5: 99, },
            }),
            False,
        ),
        (
            pd.DataFrame({
                "task": {0: "Images", 1: "Images", 2: "Images", 3: "Images", 4: "Images", 5: "Images"},
                "dprime_peak": {0: 1.2, 1: 2.0, 2: 1.1, 3: 1.1, 4: 2.0, 5: 1.2, },
                "num_engaged_trials": {0: 101, 1: 102, 2: 200, 3: 101, 4: 102, 5: 200, },
            }),
            True,
        ),
    ],
)
def test_meets_engagement_criteria(session_summary, expected):
    assert criteria.meets_engagement_criteria(session_summary) == expected


@pytest.mark.parametrize(
    "session_summary, expected",
    [
        (
            pd.DataFrame({
                "task": {0: "Images", 1: "Images", 2: "Images", 3: "Images", 4: "Images", 5: "Images"},
                "other_metric": {0: 1.2, 1: 2.0, 2: 1.1, 3: 1.1, 4: 2.0, 5: 1.2, },
                "num_engaged_trials": {0: 101, 1: 102, 2: 200, 3: 101, 4: 102, 5: 200, },
            }),
            pytest.raises(DataFrameKeyError),
        ),
        (
            pd.DataFrame({
                "task": {0: "Images", 1: "Images", },
                "dprime_peak": {0: 1.2, 1: 2.0,},
                "num_engaged_trials": {0: 101, 1: 102},
            }),
            pytest.raises(DataFrameIndexError),
        ),
    ],
)
def test_meets_engagement_criteria_exception(session_summary, expected):
    with expected:
        criteria.meets_engagement_criteria(session_summary)


@pytest.mark.parametrize(
    "trials, expected",
    [
        (pd.DataFrame({'training_day': {0: 0, 1: 1, 2: 3, }, }), False, ),
        (pd.DataFrame({'training_day': {0: 0, 1: 1, 2: 40, }, }), True, ),  # inclusive
        (pd.DataFrame({'training_day': {0: 0, 1: 1, 2: 41, }, }), True, ),
    ],
)
def test_summer_over(trials, expected):
    assert criteria.summer_over(trials) == expected
