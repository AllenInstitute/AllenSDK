import pytest
import pandas as pd

from allensdk.brain_observatory.behavior.mtrain import annotate_change_detect


@pytest.fixture
def trials():
    return pd.DataFrame({
        'trial_type': ['go', 'catch', 'go', 'catch'],
        'response':   [1.0,      1.0,   0.0,    0.0]})


def test_annotate_change_detect(trials):

    annotate_change_detect(trials)
    pd.testing.assert_series_equal(trials['change'], pd.Series([True, False, True, False], name='change'))
    pd.testing.assert_series_equal(trials['detect'], pd.Series([True, True, False, False], name='detect'))
