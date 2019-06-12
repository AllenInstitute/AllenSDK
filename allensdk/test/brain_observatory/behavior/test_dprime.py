import numpy as np

from allensdk.brain_observatory.behavior.dprime import get_hit_rate, get_false_alarm_rate, get_dprime_pipeline


NaN = float('nan')


def test_get_hit_rate():

    hit, miss, aborted = (
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0])

    result = get_hit_rate(hit=hit, miss=miss, aborted=aborted, sliding_window=3)
    np.testing.assert_allclose(result, [0, .5, 1/3, 2/3])


def test_get_false_alarm_rate():

    false_alarm, correct_reject, aborted = (
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0])

    result = get_false_alarm_rate(false_alarm=false_alarm, correct_reject=correct_reject, aborted=aborted, sliding_window=3)
    np.testing.assert_allclose(result, [0, .5, 1/3, 2/3])


def test_dprime():

    hit, miss, false_alarm, correct_reject, aborted = (
        [0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0])

    hr = get_hit_rate(hit=hit, miss=miss, aborted=aborted, sliding_window=3)
    far = get_false_alarm_rate(false_alarm=false_alarm, correct_reject=correct_reject, aborted=aborted, sliding_window=3)
    result = get_dprime_pipeline(hr, far)
    np.testing.assert_allclose(result, [NaN, NaN, NaN, 2.326348, 4.652696, 4.652696])
