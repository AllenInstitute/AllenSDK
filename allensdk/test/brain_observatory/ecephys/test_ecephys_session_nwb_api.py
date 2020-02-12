# most of the tests for this functionality are actually in test_write_nwb

import pytest
import pandas as pd

import allensdk.brain_observatory.ecephys.ecephys_session_api.ecephys_nwb_session_api as ensa


@pytest.mark.parametrize("left,right,expected,left_on,right_on", [
    [
        pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}),
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        "a",
        "a"
    ],
    [
        pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}),
        pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [7, 8, 9]}),
        pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [7, 8, 9]}),
        ["a", "b"],
        ["a", "b"]
    ]
])
def test_clobbering_merge(left, right, expected, left_on, right_on):
    obtained = ensa.clobbering_merge(left, right, left_on=left_on, right_on=left_on)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True)
