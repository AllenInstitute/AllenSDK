import pytest
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.data_objects import (
    StimulusTimestamps)
from allensdk.brain_observatory.behavior.data_objects.trials.trial import (
    Trial)


@pytest.mark.parametrize("behavior_stimuli_data_fixture, trial, expected",
                         [({},
                           {'events': [(None, None, None, 0)],
                            'stimulus_changes': []},
                           {'initial_image_name': 'gratings_90',
                            'change_image_name': 'gratings_90'}),
                          ({},
                           {'events': [(None, None, None, 0)],
                            'stimulus_changes':[
                                (('horizontal', 90),
                                 ('vertical', 180),
                                 None, None)]},
                           {'initial_image_name': 'gratings_90',
                            'change_image_name': 'gratings_180'}),
                          ({"images_set_log": [('Image', 'im065', 5, 0)],
                            "grating_set_log": [("Ori", 270, 15, 6)]},
                           {'events': [(None, None, None, 5)],
                            'stimulus_changes':
                                [(('im065', 'im065'),
                                  ('im057', 'im057'),
                                  None, None)]},
                           {'initial_image_name': 'im065',
                            'change_image_name': 'im057'})],
                         indirect=['behavior_stimuli_data_fixture'])
def test_get_trial_image_names(behavior_stimuli_data_fixture, trial,
                               expected):

    class DummyTrial(Trial):
        @staticmethod
        def _calculate_trial_end(trial_end,
                                 behavior_stimulus_file):
            return -999

        def _match_to_sync_timestamps(
                self,
                raw_stimulus_timestamps,
                licks,
                rewards,
                stimuli):
            return dict()

    stimuli = behavior_stimuli_data_fixture['items']['behavior']['stimuli']

    trial_obj = DummyTrial(trial=trial,
                           start=None,
                           end=None,
                           behavior_stimulus_file=None,
                           index=None,
                           stimulus_timestamps=None,
                           licks=None,
                           rewards=None,
                           stimuli=stimuli)

    trial_image_names = trial_obj._get_trial_image_names(stimuli)
    assert trial_image_names == expected


@pytest.mark.parametrize("behavior_stimuli_data_fixture, start_frame,"
                         "expected",
                         [({}, 0, ('grating', 90, 'gratings_90')),
                          ({
                               "images_set_log": [
                                   ('Image', 'im065', 5, 0)],
                               "grating_set_log": [
                                   ("Ori", 270, 15, 6)]}, 0,
                           ('images', 'im065', 'im065')),
                          ({
                               "images_set_log": [],
                               "grating_set_log": []
                           }, 0, ('', '', ''))],
                         indirect=['behavior_stimuli_data_fixture'])
def test_resolve_initial_image(behavior_stimuli_data_fixture, start_frame,
                               expected):

    class DummyTrial(Trial):
        @staticmethod
        def _calculate_trial_end(trial_end,
                                 behavior_stimulus_file):
            return -999

        def _match_to_sync_timestamps(
                self,
                raw_stimulus_timestamps,
                licks,
                rewards,
                stimuli):
            return dict()

    stimuli = behavior_stimuli_data_fixture['items']['behavior']['stimuli']

    trial_obj = DummyTrial(trial=None,
                           start=None,
                           end=None,
                           behavior_stimulus_file=None,
                           index=None,
                           stimulus_timestamps=None,
                           licks=None,
                           rewards=None,
                           stimuli=stimuli)

    resolved = trial_obj._resolve_initial_image(stimuli, start_frame)
    assert resolved == expected


@pytest.mark.parametrize(
    "go,catch,auto_rewarded,hit,false_alarm,aborted,errortext", [
        (False, False, False, True, False, True,
         "'aborted' trials cannot be"),  # aborted and hit
        (False, False, False, False, True, True,
         "'aborted' trials cannot be"),  # aborted and false alarm
        (False, False, True, False, False, True,
         "'aborted' trials cannot be"),  # aborted and auto_rewarded
        (False, False, False, True, True, False,
         "both `hit` and `false_alarm` cannot be True"),  # hit and false alarm
        (True, True, False, False, False, False,
         "both `go` and `catch` cannot be True"),  # go and catch
        # go and auto_rewarded
        (True, False, True, False, False, False,
         "both `go` and `auto_rewarded` cannot be True")
    ]
)
def test_get_trial_timing_exclusivity_assertions(
        go, catch, auto_rewarded, hit, false_alarm, aborted, errortext):

    # we just want to test a method of Trial, specifically test
    # errors that will be raised before any processing happens,
    # so we can define a child class with an empty __init__
    class DummyTrial(Trial):
        def __init__(self):
            pass

    with pytest.raises(AssertionError) as e:
        DummyTrial()._get_trial_timing(
            None, None, go, catch, auto_rewarded, hit, false_alarm,
            aborted)
    assert errortext in str(e.value)


def test_get_trial_timing():
    event_dict = {
        ('trial_start', ''): {'timestamp': 306.4785879253758, 'frame': 18075},
        ('initial_blank', 'enter'): {'timestamp': 306.47868008512637,
                                     'frame': 18075},
        ('initial_blank', 'exit'): {'timestamp': 306.4787637603285,
                                    'frame': 18075},
        ('pre_change', 'enter'): {'timestamp': 306.47883573270514,
                                  'frame': 18075},
        ('pre_change', 'exit'): {'timestamp': 306.4789062422286,
                                 'frame': 18075},
        ('stimulus_window', 'enter'): {'timestamp': 306.478977629464,
                                       'frame': 18075},
        ('stimulus_changed', ''): {'timestamp': 310.9827406729944,
                                   'frame': 18345},
        ('auto_reward', ''): {'timestamp': 310.98279450599154, 'frame': 18345},
        ('response_window', 'enter'): {'timestamp': 311.13223900212347,
                                       'frame': 18354},
        ('response_window', 'exit'): {'timestamp': 311.73284526699706,
                                      'frame': 18390},
        ('miss', ''): {'timestamp': 311.7330193465259, 'frame': 18390},
        ('stimulus_window', 'exit'): {'timestamp': 315.2356723770604,
                                      'frame': 18600},
        ('no_lick', 'exit'): {'timestamp': 315.23582480636213, 'frame': 18600},
        ('trial_end', ''): {'timestamp': 315.23590438557534, 'frame': 18600}
    }

    licks = [
        312.24876,
        312.58027,
        312.73126,
        312.86627,
        313.02635,
        313.16292,
        313.54016,
        314.04408,
        314.47449,
        314.61011,
        314.75495,
    ]

    # Only need to worry about the timestamp
    # value at change_frame
    # because get_trial_timing will only use
    # timestamps to lookup the timestamp of
    # change_frame
    timestamps = np.zeros(20000, dtype=float)
    timestamps[18345] = 311.77086
    monitor_delay = 0.01

    # need a mock Trial class that just populates
    # self._stimulus_timestamps
    stimulus_timestamps = StimulusTimestamps(
        timestamps=timestamps,
        monitor_delay=monitor_delay)

    class DummyTrial(Trial):
        def __init__(self, timestamps):
            self._stimulus_timestamps = timestamps

    this_trial = DummyTrial(timestamps=stimulus_timestamps)

    result = this_trial._get_trial_timing(
        event_dict,
        licks,
        go=False,
        catch=False,
        auto_rewarded=True,
        hit=False,
        false_alarm=False,
        aborted=False
    )

    expected_result = {
        'start_time': 306.4785879253758,
        'stop_time': 315.23590438557534,
        'trial_length': 8.757316460199547,
        'response_time': 312.24876,
        'change_frame': 18345,
        'change_time': 311.78086,
        'response_latency': 0.4678999999999769
    }

    # use assert_frame_equal to take advantage of the
    # nice way it deals with NaNs
    pd.testing.assert_frame_equal(pd.DataFrame(result, index=[0]),
                                  pd.DataFrame(expected_result, index=[0]),
                                  check_names=False)
