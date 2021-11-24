import numpy as np
import pytest

from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df, calc_deriv, deg_to_dist, _shift, _identify_wraps,
    _unwrap_voltage_signal, _angular_change, _zscore_threshold_1d,
    _clip_speed_wraps)

import allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing as rp  # noqa: E501


@pytest.fixture
def timestamps():
    return np.arange(0., 10., 0.1)


@pytest.fixture
def running_data():
    rng = np.random.default_rng()
    return {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": rng.random((100,)),
                        "vsig": rng.uniform(low=0.0, high=5.1, size=(100,)),
                        "vin": rng.uniform(low=4.9, high=5.0, size=(100,)),
                    }]}}}


@pytest.mark.parametrize(
    "x,time,expected", [
        ([1.0, 1.0], [1.0, 2.0], [np.nan, 0.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [np.nan, 1.0, 1.0]),
        ([1.0, 2.0, 3.0], [1.0, 4.0, 6.0], [np.nan, 1.0/3.0, 1.0/2.0])
    ]
)
def test_calc_deriv(x, time, expected):
    obtained = calc_deriv(x, time)

    # np.nan == np.nan returns False so filter out the first values
    obtained = obtained[1:]
    expected = expected[1:]

    assert np.all(obtained == expected)


@pytest.mark.parametrize(
    "speed,expected", [
        (np.array([1.0]), [5.5033]),
        (np.array([0., 2.0]), [0., 11.0066])
    ]
)
def test_deg_to_dist(speed, expected):
    np.testing.assert_allclose(deg_to_dist(speed), expected, atol=0.0001)


@pytest.mark.parametrize(
    "lowpass", [True, False]
)
def test_get_running_df(running_data, timestamps, lowpass):
    actual = get_running_df(running_data, timestamps, lowpass=lowpass)
    np.testing.assert_array_equal(actual.index, timestamps)
    assert sorted(list(actual)) == ["dx", "speed", "v_in", "v_sig"]
    # Should bring raw data through
    np.testing.assert_array_equal(
        actual["v_sig"].values,
        running_data["items"]["behavior"]["encoders"][0]["vsig"])
    np.testing.assert_array_equal(
        actual["v_in"].values,
        running_data["items"]["behavior"]["encoders"][0]["vin"])
    np.testing.assert_array_equal(
        actual["dx"].values,
        running_data["items"]["behavior"]["encoders"][0]["dx"])
    if lowpass:
        assert np.count_nonzero(np.isnan(actual["speed"])) == 0


@pytest.mark.parametrize(
    "lowpass", [True, False]
)
def test_get_running_df_one_fewer_timestamp_check_warning(running_data,
                                                          timestamps,
                                                          lowpass):
    with pytest.warns(
        UserWarning,
        match="Time array is 1 value shorter than encoder array.*"
    ):
        # Call with one fewer timestamp, check for a warning
        _ = get_running_df(
            data=running_data,
            time=timestamps[:-1],
            lowpass=lowpass
        )


@pytest.mark.parametrize(
    "lowpass", [True, False]
)
def test_get_running_df_one_fewer_timestamp_check_truncation(running_data,
                                                             timestamps,
                                                             lowpass):
    # Call with one fewer timestamp
    output = get_running_df(
        data=running_data,
        time=timestamps[:-1],
        lowpass=lowpass
    )

    # Check that the output is actually trimmed, and the values are the same
    assert len(output) == len(timestamps) - 1
    np.testing.assert_equal(
        output["v_sig"],
        running_data["items"]["behavior"]["encoders"][0]["vsig"][:-1]
    )
    np.testing.assert_equal(
        output["v_in"],
        running_data["items"]["behavior"]["encoders"][0]["vin"][:-1]
    )


@pytest.mark.parametrize(
    "arr, periods, fill, expected",
    [
        ([1, 2, 3], 1, None, np.array([np.nan, 1., 2.])),
        ([1, 2, 3], 2, 99., np.array([99., 99., 1.])),
        ([1, 2, 3], 3, 99., np.array([99., 99., 99.])),
        ([1, 2, 3], 4, 99., np.array([99., 99., 99.])),
        ([], 2, 30, np.array([]))
    ]
)
def test_shift(arr, periods, fill, expected):
    actual = _shift(arr, periods, fill)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "periods", [0, -2]
)
def test_shift_raises_error_periods_zero(periods):
    with pytest.raises(ValueError, match="Can only shift"):
        _shift(np.ones((5,)), periods)


@pytest.mark.parametrize(
    "arr, min_threshold, max_threshold, expected",
    [
        (np.array(
            [0, 2, 5, 0,    # pos wrap 5-0
             2, 0, 5    # neg wrap 0-5
             ]), 1.5, 3.5, (np.array([3]), np.array([6]))),
        (np.array([0, 2, 5, 0, 2, 5]), 0, 5, (np.array([]), np.array([]))),
    ]
)
def test_identify_wraps(arr, min_threshold, max_threshold, expected):
    actual = _identify_wraps(
        arr, min_threshold=min_threshold, max_threshold=max_threshold)
    np.testing.assert_array_equal(
        actual[0], expected[0],
        f"error identifying positive wraps, got {actual[0]}, "
        f"expected {expected[0]}")
    np.testing.assert_array_equal(
        actual[1], expected[1],
        f"error identifying negative wraps, got {actual[1]}, "
        f"expected {expected[1]}")


@pytest.mark.parametrize(
    "vsig, pos_wrap_ix, neg_wrap_ix, vmax, max_threshold, max_diff, expected",
    [
        (   # No artifacts or baseline
            np.array([0, 1, 3, 5, 0.5, 1, 2.5, 5, 0, 1, 4, 3]),
            np.array([4, 8]), np.array([10]), 5.0, 5.1, 3.0,
            np.array([np.nan, 1, 3, 5, 5.5, 6, 7.5, 10, 10, 11, 9, 8])
        ),
        (   # Some diff artifacts, baseline
            np.array([1, 1, 3, 5, 0.5, 1, 2.5, 5, 0, 1, 4, 1.5]),
            np.array([4, 8]), np.array([10]), 5.0, 5.1, 2.0,
            np.array([np.nan, 1, 3, 5, 5.5, 6, 7.5, np.nan, 7.5, 8.5, 6.5,
                      np.nan])
        ),
        (   # Max artifact -- use threshold instead
            np.array([0, 7, 3, 5, 0.5, 1]),
            np.array([2, 4]), np.array([]).astype(int), None, 5.1, 6.0,
            np.array([np.nan, np.nan, 1, 3, 3.5, 4])
        ),
        (
            # No wraps
            np.ones(5,), np.array([]), np.array([]), 5.0, 5.1, 3.0,
            np.array([np.nan, 1., 1., 1., 1.])
        )
    ]
)
def test_unwrap_voltage_signal(
        vsig, pos_wrap_ix, neg_wrap_ix, vmax, max_threshold, max_diff,
        expected):
    actual = _unwrap_voltage_signal(
        vsig, pos_wrap_ix, neg_wrap_ix, vmax=vmax,
        max_threshold=max_threshold, max_diff=max_diff)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "vsig, vmax, expected",
    [
        (
            np.array([1, 2, 3, 4, 5]), 2.0,
            np.array([np.nan, np.pi, np.pi, np.pi, np.pi])
        ),
        (
            np.array([np.nan, 1, 3, np.nan, 4]),
            np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            np.array([np.nan, np.nan, 2*np.pi, np.nan, np.nan]),
        )
    ]
)
def test_angular_change(vsig, vmax, expected):
    actual = _angular_change(vsig, vmax)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "arr, threshold, expected",
    [
        (np.ones(5,), 2.0, np.ones(5,)),
        (np.array([99, 1, np.nan, 1, 1, 1]), 1.5,
         np.array([np.nan, 1, np.nan, 1, 1, 1])),
        (np.ones(1,), 2.0, np.ones(1,))
    ]
)
def test_zscore_threshold_1d(arr, threshold, expected):
    actual = _zscore_threshold_1d(arr, threshold=threshold)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "speed, time, wrap_indices, span, expected",
    [
        (   # Clip bottom, then clip top, then no clip required
            np.array([0, 0, -1, 5, 0, 99, 6, 1, 2, 3]),
            np.array(range(10)).astype(float),
            [2, 5, 8],
            1.0,
            np.array([0, 0, 0, 5, 0, 6, 6, 1, 2, 3])
        ),
    ]
)
def test_clip_speed_wraps(
        speed, time, wrap_indices, span, expected, monkeypatch):
    monkeypatch.setattr(rp, "_local_boundaries", lambda x, y, z: (y-1, y+1))
    actual = _clip_speed_wraps(speed, time, wrap_indices, span)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "time, index, span",
    [
        (np.arange(10.), 0, 2.0),    # no neighborhood before first point
        (np.arange(10.), 9, 2.0),    # no neighborhood after last point
        (np.arange(10.), 4, 0.25),   # data not sampled with enough frequency
    ]
)
def test_local_boundaries_raises_warning(time, index, span):
    with pytest.warns(UserWarning, match="Unable to find"):
        rp._local_boundaries(time, index, span)


@pytest.mark.parametrize(
    "time, index",
    [
        (np.array([5., 4., 3., 4., 5.]), 2),
        (np.array([1., 2., 3., 2., 1.]), 2),
        (np.array([3., 3., 3., 2., 1.]), 2),
    ]
)
def test_local_boundaries_raises_error_non_monotonic(time, index):
    with pytest.raises(ValueError, match="Data do not monotonically"):
        rp._local_boundaries(time, index, 1.0)


@pytest.mark.parametrize(
    "time, index, span, expected",
    [
        (np.arange(10.), 4, 2.0, (2.0, 6.0)),   # Spans > 1 element +/-
        (np.arange(10.), 2, 1.0, (1.0, 3.0))    # Spans = 1 element +/-
    ]
)
def test_local_boundaries(time, index, span, expected):
    actual = rp._local_boundaries(time, index, span)
    assert expected == actual
