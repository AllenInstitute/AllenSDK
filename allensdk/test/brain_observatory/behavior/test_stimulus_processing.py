import pandas as pd
import numpy as np
import pytest

from allensdk.brain_observatory.behavior.stimulus_processing import (
    get_stimulus_presentations, _get_stimulus_epoch, _get_draw_epochs,
    get_visual_stimuli_df, get_stimulus_metadata, get_gratings_metadata,
    get_stimulus_templates)


@pytest.fixture()
def behavior_stimuli_time_fixture(request):
    """
    Fixture that allows for parameterization of behavior_stimuli stimuli
    time data.
    """
    timestamp_count = request.param["timestamp_count"]
    time_step = request.param["time_step"]

    timestamps = np.array([time_step * i for i in range(timestamp_count)])

    return timestamps


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
            ("Ori", 90.0, 3.585, 0),
            ("Ori", 180.0, 40.847, 6),
            ("Ori", 270.0, 62.633, 12)],
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


@pytest.mark.parametrize("behavior_stimuli_data_fixture, remove_stimuli, "
                         "expected_templates",
                         [({}, ['images'], {})],
                         indirect=["behavior_stimuli_data_fixture"])
def test_get_stimulus_templates(behavior_stimuli_data_fixture, remove_stimuli,
                                expected_templates):
    for stimuli in remove_stimuli:
        del (behavior_stimuli_data_fixture['items']['behavior']
        ['stimuli'][stimuli])
    templates = get_stimulus_templates(behavior_stimuli_data_fixture)
    assert templates == expected_templates


# def test_get_images_dict():
#     pass
#     # TODO
#     # This is too hard-coded to be testable right now.
#     # convert_filepath_caseinsensitive prevents using any tempdirs/tempfiles


@pytest.mark.parametrize("behavior_stimuli_data_fixture, remove_stimuli, "
                         "starting_index, expected_metadata", [
                             ({
                                  "grating_set_log": []
                              }, [], 0,
                              {
                                  'image_category': {},
                                  'image_name': {},
                                  'image_set': {},
                                  'phase': {},
                                  'spatial_frequency': {},
                                  'orientation': {},
                                  'image_index': {}
                              }),
                             ({}, [], 0,
                              {
                                  'image_category': {0: 'grating'},
                                  'image_name': {0: 'gratings_90.0'},
                                  'image_set': {0: 'grating'},
                                  'phase': {0: None},
                                  'spatial_frequency': {0: None},
                                  'orientation': {0: 90},
                                  'image_index': {0: 0}
                              }),
                             ({'grating_phase': 0.5,
                               'grating_spatial_frequency': 12}, [], 0,
                              {
                                  'image_category': {0: 'grating'},
                                  'image_name': {0: 'gratings_90.0'},
                                  'image_set': {0: 'grating'},
                                  'phase': {0: 0.5},
                                  'spatial_frequency': {0: 12},
                                  'orientation': {0: 90},
                                  'image_index': {0: 0}
                              }),
                             ({"grating_set_log": [
                                 ("Ori", 90.0, 3.5, 0),
                                 ("Ori", 270.0, 15, 6)],
                                  "grating_phase": 0.5,
                                  "grating_spatial_frequency": 12},
                              [], 12,
                              {
                                  'image_category': {0: 'grating',
                                                     1: 'grating'},
                                  'image_name': {0: 'gratings_90.0',
                                                 1: 'gratings_270.0'},
                                  'image_set': {0: 'grating',
                                                1: 'grating'},
                                  'phase': {0: 0.5, 1: 0.5},
                                  'spatial_frequency': {0: 12, 1: 12},
                                  'orientation': {0: 90, 1: 270},
                                  'image_index': {0: 12, 1: 13}
                              }),
                             ({}, ['grating'], 0,
                              {
                                  'image_category': {},
                                  'image_name': {},
                                  'image_set': {},
                                  'phase': {},
                                  'spatial_frequency': {},
                                  'orientation': {},
                                  'image_index': {}
                              }),
                             ({"grating_set_log":
                                 [
                                     ("Ori", 90, 3, 0)],
                                  "grating_phase": 0.5,
                                  "grating_spatial_frequency": 0.25},
                              [], 0,
                              {
                                  'image_category': {0: 'grating'},
                                  'image_name': {0: 'gratings_90.0'},
                                  'image_set': {0: 'grating'},
                                  'phase': {0: 0.5},
                                  'spatial_frequency': {0: 0.25},
                                  'orientation': {0: 90},
                                  'image_index': {0: 0}
                              })
                         ],
                         indirect=['behavior_stimuli_data_fixture'])
def test_get_gratings_metadata(behavior_stimuli_data_fixture, remove_stimuli,
                               starting_index, expected_metadata):
    stimuli = behavior_stimuli_data_fixture['items']['behavior']['stimuli']
    for remove_stim in remove_stimuli:
        del stimuli[remove_stim]
    grating_meta = get_gratings_metadata(stimuli, start_idx=starting_index)

    assert grating_meta.to_dict() == expected_metadata


@pytest.mark.parametrize("behavior_stimuli_data_fixture, remove_stimuli, "
                         "expected_metadata", [
                             ({'grating_phase': 10.0,
                               'grating_spatial_frequency': 90.0,
                               "grating_set_log": [
                                   ("Ori", 90.0, 3.585, 0),
                                   ("Ori", 180.0, 40.847, 6),
                                   ("Ori", 270.0, 62.633, 12)]
                               },
                              ['images'],
                              {'image_index': [0, 1, 2, 3],
                               'image_name': ['gratings_90.0',
                                              'gratings_180.0',
                                              'gratings_270.0', 'omitted'],
                               'image_category': ['grating',
                                                  'grating', 'grating',
                                                  'omitted'],
                               'image_set': ['grating', 'grating',
                                             'grating', 'omitted'],
                               'phase': [10, 10, 10, None],
                               'spatial_frequency': [90, 90,
                                             90, None],
                               'orientation': [90, 180, 270, None]}),
                             ({}, ['images', 'grating'],
                              {'image_index': [0],
                               'image_name': ['omitted'],
                               'image_category': ['omitted'],
                               'image_set': ['omitted'],
                               'phase': [None],
                               'spatial_frequency': [None],
                               'orientation': [None]})],
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
                              {"duration": [3.0, 2.0, 3.0, 2.0],
                               "end_frame": [5.0, 5.0, 12.0, 12.0],
                               "image_name": [np.NaN, 'im065', np.NaN,
                                              'im064'],
                               "index": [2, 0, 3, 1],
                               "omitted": [False, False, False, False],
                               "orientation": [90, np.NaN, 270, np.NaN],
                               "start_frame": [2.0, 3.0, 9.0, 10.0],
                               "start_time": [2, 3, 9, 10],
                               "stop_time": [5, 5, 12, 12]})
                         ], indirect=['behavior_stimuli_time_fixture',
                                      'behavior_stimuli_data_fixture'])
def test_get_stimulus_presentations(behavior_stimuli_time_fixture,
                                    behavior_stimuli_data_fixture,
                                    expected):
    presentations_df = get_stimulus_presentations(
        behavior_stimuli_data_fixture,
        behavior_stimuli_time_fixture)

    expected_df = pd.DataFrame.from_dict(expected)

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
    stimuli_df = get_visual_stimuli_df(behavior_stimuli_data_fixture,
                                       behavior_stimuli_time_fixture)
    stimuli_df = stimuli_df.drop('index', axis=1)

    expected_df = pd.DataFrame.from_dict(expected_data)
    assert stimuli_df.equals(expected_df)
