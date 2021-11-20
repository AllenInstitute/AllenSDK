import numpy as np
import pytest
import pandas as pd
import datetime
import pytz

from allensdk.brain_observatory.behavior.dprime import get_hit_rate, get_false_alarm_rate, get_rolling_dprime, get_trial_count_corrected_false_alarm_rate, get_trial_count_corrected_hit_rate, get_dprime


NaN = float('nan')


@pytest.fixture
def mock_trials_fixture():

    n_tr = 500
    np.random.seed(42)
    change = np.random.random(n_tr) > 0.8
    incorrect = np.random.random(n_tr) > 0.8
    detect = change.copy()
    detect[incorrect] = ~detect[incorrect]

    trials = pd.DataFrame({
        'change': change,
        'detect': detect,
    },)
    trials['trial_type'] = trials['change'].map(lambda x: ['catch', 'go'][x])
    trials['response'] = trials['detect']
    trials['change_time'] = np.sort(np.random.rand(n_tr)) * 3600
    trials['reward_lick_latency'] = 0.1
    trials['reward_lick_count'] = 10
    trials['auto_rewarded'] = False
    trials['lick_frames'] = [[] for row in trials.iterrows()]
    trials['trial_length'] = 8.5
    trials['reward_times'] = trials.apply(lambda r: [r['change_time']+0.2] if r['change']*r['detect'] else [],axis=1)
    trials['reward_volume'] = 0.005 * trials['reward_times'].map(len)
    trials['response_latency'] = trials.apply(lambda r: 0.2 if r['detect'] else np.inf,axis=1)
    trials['blank_duration_range'] = [[0.5, 0.5] for row in trials.iterrows()]

    metadata = {}
    metadata['mouse_id'] = 'M999999'
    metadata['user_id'] = 'johnd'

    metadata['startdatetime'] = datetime.datetime(2017, 7, 19, 10, 35, 8, 369000, tzinfo=pytz.utc)
    metadata['dayofweek'] = metadata['startdatetime'].weekday()
    metadata['startdatetime'] = metadata['startdatetime']

    metadata['behavior_session_uuid'] = 12345
    metadata['stage'] = 'test'
    metadata['stimulus'] = 'natural_scenes'
    metadata['stimulus_distribution'] = 'exponential'

    for k, v in metadata.items():
        trials[k] = v
    return trials

from collections import defaultdict

@pytest.fixture
def mock_rolling_dprime_fixture(mock_trials_fixture):

    data_dict = defaultdict(list)
    for ri, row in mock_trials_fixture[['trial_type', 'response', 'change_time']].iterrows():
        assert not pd.isnull(row['change_time'])
        if row['trial_type'] == 'go' and row['response'] == True:
            hit = True
            miss = false_alarm = correct_reject = False
        elif row['trial_type'] == 'go' and row['response'] == False:
            miss = True
            hit = false_alarm = correct_reject = False
        elif row['trial_type'] == 'catch' and row['response'] == True:
            false_alarm = True
            miss = hit = correct_reject = False
        elif row['trial_type'] == 'catch' and row['response'] == False:
            correct_reject = True
            hit = false_alarm = miss = False
        else:
            raise RuntimeError
        data_dict['hit'].append(hit)
        data_dict['miss'].append(miss)
        data_dict['false_alarm'].append(false_alarm)
        data_dict['correct_reject'].append(correct_reject)
        data_dict['aborted'].append(False)



    return pd.DataFrame(data_dict)

def test_get_hit_rate():

    hit, miss, aborted = (
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0])

    result = get_hit_rate(hit=hit, miss=miss, aborted=aborted, sliding_window=3)
    np.testing.assert_allclose(result, [0, .5, 1/3, 2/3])


def test_get_false_alarm_rate(mock_trials_fixture):

    false_alarm, correct_reject, aborted = (
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0])

    result = get_false_alarm_rate(false_alarm=false_alarm, correct_reject=correct_reject, aborted=aborted, sliding_window=3)
    np.testing.assert_allclose(result, [0, .5, 1/3, 2/3])


def test_rolling_dprime_unit():

    hit, miss, false_alarm, correct_reject, aborted = (
        [0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0])

    hr = get_hit_rate(hit=hit, miss=miss, aborted=aborted, sliding_window=3)
    far = get_false_alarm_rate(false_alarm=false_alarm, correct_reject=correct_reject, aborted=aborted, sliding_window=3)
    result = get_rolling_dprime(hr, far)
    np.testing.assert_allclose(result, [NaN, NaN, NaN, 2.326348, 4.652696, 4.652696])


def test_rolling_dprime_integration_legacy(mock_rolling_dprime_fixture):
    sliding_window = 100

    hit = mock_rolling_dprime_fixture.hit
    miss = mock_rolling_dprime_fixture.miss
    false_alarm = mock_rolling_dprime_fixture.false_alarm
    correct_reject = mock_rolling_dprime_fixture.correct_reject
    aborted = mock_rolling_dprime_fixture.aborted

    hr = get_hit_rate(hit=hit, miss=miss, aborted=aborted, sliding_window=sliding_window)
    cr = get_false_alarm_rate(false_alarm=false_alarm, correct_reject=correct_reject, aborted=aborted, sliding_window=sliding_window)
    dprime = get_rolling_dprime(hr, cr)

    assert dprime[2] == 4.6526957480816815


def test_rolling_dprime_integration(mock_rolling_dprime_fixture):
    sliding_window = 100

    hit = mock_rolling_dprime_fixture.hit
    miss = mock_rolling_dprime_fixture.miss
    false_alarm = mock_rolling_dprime_fixture.false_alarm
    correct_reject = mock_rolling_dprime_fixture.correct_reject
    aborted = mock_rolling_dprime_fixture.aborted

    hr = get_trial_count_corrected_hit_rate(hit=hit, miss=miss, aborted=aborted, sliding_window=sliding_window)
    cr = get_trial_count_corrected_false_alarm_rate(false_alarm=false_alarm, correct_reject=correct_reject, aborted=aborted, sliding_window=sliding_window)
    dprime = get_rolling_dprime(hr, cr)

    assert dprime[2] == 0.6744897501960817


@pytest.mark.parametrize('hr, far, dprime', [
    pytest.param(1., 1., 0.),
    pytest.param(.5, .5, 0.),
    pytest.param(.25, .5, -0.6744897501960817),
    pytest.param(.5, .25, 0.6744897501960817),
])
def test_dprime(hr, far, dprime):
    val = get_dprime(hr, far)
    assert val == dprime
    if hr == far:
        assert dprime == 0
    if hr < far:
        assert val < 0
    elif hr > far:
        assert val > 0
    else:
        pass

