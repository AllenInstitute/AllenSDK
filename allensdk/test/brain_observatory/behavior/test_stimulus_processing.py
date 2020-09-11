import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.stimulus_processing import (
    get_stimulus_presentations, _get_stimulus_epoch, _get_draw_epochs,
    get_visual_stimuli_df, get_stimulus_metadata)


@pytest.mark.parametrize(
    "behavior_stimuli_data_fixture,current_set_ix,start_frame,"
    "n_frames,expected", [
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 0, 18, (0, 6)),
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         2, 11, 18, (11, 18))
    ], indirect=["behavior_stimuli_data_fixture"]
)
def test_get_stimulus_epoch(behavior_stimuli_data_fixture,
                            current_set_ix, start_frame, n_frames, expected):
    log = (behavior_stimuli_data_fixture["items"]["behavior"]["stimuli"]
    ["images"]["set_log"])
    actual = _get_stimulus_epoch(log, current_set_ix, start_frame, n_frames)
    assert actual == expected


@pytest.mark.parametrize(
    "behavior_stimuli_data_fixture,start_frame,stop_frame,expected,"
    "stimuli_type", [
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 6, [(1, 4)], 'images'),
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 11, [(1, 4), (8, 11)], 'images'),
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 22, [(1, 4), (8, 11), (15, 18)], 'images'),
        ({"grating_set_log": [
            ("Ori", 90, 3.585, 0),
            ("Ori", 180, 40.847, 6),
            ("Ori", 270, 62.633, 12)],
             "grating_draw_log": ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 6, [(1, 4)], 'grating'),
        ({"grating_set_log": [
            ("Ori", 90, 3.585, 0),
            ("Ori", 180, 40.847, 6),
            ("Ori", 270, 62.633, 12)],
             "grating_draw_log": ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         6, 11, [(8, 11)], 'grating')
    ], indirect=['behavior_stimuli_data_fixture']
)
def test_get_draw_epochs(behavior_stimuli_data_fixture,
                         start_frame, stop_frame, expected, stimuli_type):
    draw_log = (behavior_stimuli_data_fixture["items"]["behavior"]
    ["stimuli"][stimuli_type]["draw_log"])  # noqa: E128
    actual = _get_draw_epochs(draw_log, start_frame, stop_frame)
    assert actual == expected


# def test_get_stimulus_templates():
#     pass
#     # TODO
#     # See below (get_images_dict is a dependency)


# def test_get_images_dict():
#     pass
#     # TODO
#     # This is too hard-coded to be testable right now.
#     # convert_filepath_caseinsensitive prevents using any tempdirs/tempfiles


@pytest.mark.parametrize("behavior_stimuli_data_fixture, remove_stimuli, "
                         "expected_metadata", [
                             ({'grating_phase': 10.0,
                               'grating_correct_frequency': 90.0},
                              ['images'],
                              {'image_index': [0, 1, 2, 3, 4],
                               'image_name': ['gratings_0.0', 'gratings_90.0',
                                              'gratings_180.0',
                                              'gratings_270.0', 'omitted'],
                               'image_category': ['grating', 'grating',
                                                  'grating', 'grating',
                                                  'omitted'],
                               'image_set': ['grating', 'grating', 'grating',
                                             'grating', 'omitted'],
                               'phase': [10, 10, 10, 10, None],
                               'correct_frequency': [90, 90, 90,
                                                     90, None]}),
                             ({}, ['images', 'grating'],
                              {'image_index': [0],
                               'image_name': ['omitted'],
                               'image_category': ['omitted'],
                               'image_set': ['omitted'],
                               'phase': [None],
                               'correct_frequency': [None]})],
                         indirect=['behavior_stimuli_data_fixture'])
def test_get_stimulus_metadata(behavior_stimuli_data_fixture,
                               remove_stimuli, expected_metadata):
    for key in remove_stimuli:
        # do this because at current images are not tested and there's a
        # hard coded path that prevents testing when this is fixed this can
        # be removed.
        del behavior_stimuli_data_fixture['items']['behavior']['stimuli'][key]
    stimulus_metadata = get_stimulus_metadata(behavior_stimuli_data_fixture)

    expected_df = pd.DataFrame.from_dict(expected_metadata)
    expected_df.set_index(['image_index'], inplace=True, drop=True)

    assert stimulus_metadata.equals(expected_df)


@pytest.mark.parametrize("behavior_stimuli_time_fixture,"
                         "behavior_stimuli_data_fixture, "
                         "expected", [
                             ({"timestamp_count": 15, "time_step": 1},
                              {"images_set_log": [
                                  ('Image', 'im065', 5, 0),
                                  ('Image', 'im064', 25, 6)
                              ],
                                  "images_draw_log": (([0] * 2 + [1] * 2 +
                                                       [0] * 3) * 2 + [0]),
                                  "grating_set_log": [
                                      ("Ori", 90, 3.5, 0),
                                      ("Ori", 270, 15, 6)
                                  ],
                                  "grating_draw_log": (([0] + [1] * 3 + [0] * 3)
                                                       * 2 + [0])},
                              {"orientation": [90, None, 270, None],
                               "image_name": [None, 'im065', None, 'im064'],
                               "start_frame": [2.0, 3.0, 9.0, 10.0],
                               "end_frame": [5.0, 5.0, 12.0, 12.0],
                               "start_time": [2, 3, 9, 10],
                               "stop_time": [5, 5, 12, 12],
                               "duration": [3.0, 2.0, 3.0, 2.0],
                               "omitted": [False, False, False, False],
                               "stimulus_presentations_id": [0, 1, 2, 3]})
                         ], indirect=['behavior_stimuli_time_fixture',
                                      'behavior_stimuli_data_fixture'])
def test_get_stimulus_presentations(behavior_stimuli_time_fixture,
                                    behavior_stimuli_data_fixture,
                                    expected):
    (behavior_stimuli_times,
     behavior_fixture_params) = behavior_stimuli_time_fixture
    presentations_df = get_stimulus_presentations(
        behavior_stimuli_data_fixture,
        behavior_stimuli_times)

    presentations_df = presentations_df.drop('index', axis=1)

    expected_df = pd.DataFrame.from_dict(expected)
    expected_df.set_index('stimulus_presentations_id', inplace=True)
    expected_df = expected_df[sorted(expected_df.columns)]

    assert presentations_df.equals(expected_df)


@pytest.mark.parametrize("behavior_stimuli_time_fixture,"
                         "behavior_stimuli_data_fixture,"
                         "expected_data", [
                             ({"timestamp_count": 15, "time_step": 1},
                              {"images_set_log": [
                                  ('Image', 'im065', 5, 0),
                                  ('Image', 'im064', 25, 6)
                              ],
                                  "images_draw_log": (([0] * 2 + [1] * 2 +
                                                       [0] * 3) * 2 + [0]),
                                  "grating_set_log": [
                                      ("Ori", 90, 3.5, 0),
                                      ("Ori", 270, 15, 6)
                                  ],
                                  "grating_draw_log": (([0] + [1] * 3 + [0] * 3)
                                                       * 2 + [0])},
                              {"orientation": [90, None, 270, None],
                               "image_name": [None, 'im065', None, 'im064'],
                               "frame": [2.0, 3.0, 9.0, 10.0],
                               "end_frame": [5.0, 5.0, 12.0, 12.0],
                               "time": [2.0, 3.0, 9.0, 10.0],
                               "duration": [3.0, 2.0, 3.0, 2.0],
                               "omitted": [False, False, False, False]})
                         ],
                         indirect=["behavior_stimuli_time_fixture",
                                   "behavior_stimuli_data_fixture"])
def test_get_visual_stimuli_df(behavior_stimuli_time_fixture,
                               behavior_stimuli_data_fixture,
                               expected_data):
    (behavior_stimuli_times,
     behavior_fixture_params) = behavior_stimuli_time_fixture
    stimuli_df = get_visual_stimuli_df(behavior_stimuli_data_fixture,
                                       behavior_stimuli_times)
    stimuli_df = stimuli_df.drop('index', axis=1)

    expected_df = pd.DataFrame.from_dict(expected_data)
    assert stimuli_df.equals(expected_df)
