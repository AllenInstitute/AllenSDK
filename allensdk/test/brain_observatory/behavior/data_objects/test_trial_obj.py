import pytest
from allensdk.brain_observatory.behavior.data_objects.trials.trial import (
    Trial)


@pytest.mark.parametrize("behavior_stimuli_data_fixture, trial, expected",
                         [({},
                           {
                               'events':
                                   [
                                       (None, None, None, 0)
                                   ],
                               'stimulus_changes':
                                   [
                                   ]
                           },
                           {
                               'initial_image_name': 'gratings_90',
                               'change_image_name': 'gratings_90'
                           }),
                          ({},
                           {
                               'events':
                                   [
                                       (None, None, None, 0)
                                   ],
                               'stimulus_changes':
                                   [
                                       (('horizontal', 90),
                                        ('vertical', 180),
                                        None,
                                        None)
                                   ]
                           },
                           {
                               'initial_image_name': 'gratings_90',
                               'change_image_name': 'gratings_180'
                           }),
                           ({
                              "images_set_log": [
                                  ('Image', 'im065', 5, 0)],
                              "grating_set_log": [
                                  ("Ori", 270, 15, 6)]
                          },
                           {
                               'events':
                                   [
                                       (None, None, None, 5)
                                   ],
                               'stimulus_changes':
                                   [
                                       (('im065', 'im065'), ('im057', 'im057'),
                                        None, None)
                                   ]
                           },
                           {
                               'initial_image_name': 'im065',
                               'change_image_name': 'im057'
                           }
                         )],
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
                stimulus_timestamps,
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
                stimulus_timestamps,
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


