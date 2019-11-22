
import numpy as np
import pytest


from allensdk.brain_observatory.behavior.stimulus_processing import (
    get_stimulus_presentations, _get_stimulus_epoch, _get_draw_epochs)


data = {
    "items": {
        "behavior": {
            "stimuli": {
                "images": {
                    "set_log": [
                        ('Image', 'im065', 5.809955710916157, 0),
                        ('Image', 'im061', 314.06612555068784, 6),
                        ('Image', 'im062', 348.5941232265203, 12),
                    ],
                    "draw_log": ([0]+[1]*3 + [0]*3)*3 + [0]
                }
            },
            # "intervalsms": np.array([16.0]*10)
        }
    }
}
timestamps = np.array([0.016 * i for i in range(19)])


@pytest.mark.parametrize(
    "current_set_ix,start_frame,n_frames,expected", [
        (0, 0, 18, (0, 6)),
        (2, 11, 18, (11, 18))
    ]
)
def test_get_stimulus_epoch(current_set_ix, start_frame, n_frames, expected):
    log = data["items"]["behavior"]["stimuli"]["images"]["set_log"]
    actual = _get_stimulus_epoch(log, current_set_ix, start_frame, n_frames)
    assert actual == expected


@pytest.mark.parametrize(
    "start_frame,stop_frame,expected", [
        (0, 6, [(1, 4)]),
        (0, 11, [(1, 4), (8, 11)])
    ]
)
def test_get_draw_epochs(start_frame, stop_frame, expected):
    draw_log = data["items"]["behavior"]["stimuli"]["images"]["draw_log"]
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


# def test_get_stimulus_presentations():
#     pass
#     # TODO
#     # Monster function with undocumented dependencies


# def test_get_visual_stimuli_df():
#     pass
#     # TODO
#     # See above (this is a dependency of get_stimulus_presentations)
