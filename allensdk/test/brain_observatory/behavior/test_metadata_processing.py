import pytest
import numpy as np

from allensdk.brain_observatory.behavior.metadata_processing import (
    OPHYS_1_3_DESCRIPTION, OPHYS_2_DESCRIPTION, OPHYS_4_6_DESCRIPTION,
    OPHYS_5_DESCRIPTION, get_task_parameters, get_expt_description)


def test_get_task_parameters():
    data = {
        "items": {
            "behavior": {
                "config": {
                    "DoC": {
                        "blank_duration_range": (0.5, 0.6),
                        "stimulus_window": 6.0,
                        "response_window": [0.15, 0.75],
                        "change_time_dist": "geometric",
                    },
                    "reward": {
                        "reward_volume": 0.007,
                    },
                    "behavior": {
                        "task_id": "DoC_untranslated",
                    },
                },
                "params": {
                    "stage": "TRAINING_3_images_A",
                },
                "stimuli": {
                    "images": {"draw_log": [1]*10}
                },
            }
        }
    }
    actual = get_task_parameters(data)
    expected = {
        "blank_duration_sec": [0.5, 0.6],
        "stimulus_duration_sec": 6.0,
        "omitted_flash_fraction": np.nan,
        "response_window_sec": [0.15, 0.75],
        "reward_volume": 0.007,
        "stage": "TRAINING_3_images_A",
        "stimulus": "images",
        "stimulus_distribution": "geometric",
        "task": "DoC_untranslated",
        "n_stimulus_frames": 10
    }
    for k, v in actual.items():
        # Special nan checking since pytest doesn't do it well
        try:
            if np.isnan(v):
                assert np.isnan(expected[k])
            else:
                assert expected[k] == v
        except (TypeError, ValueError):
            assert expected[k] == v
    assert list(actual.keys()) == list(expected.keys())


@pytest.mark.parametrize("session_type, expected_description", [
    ("OPHYS_1_images_A", OPHYS_1_3_DESCRIPTION),
    ("OPHYS_2_images_B", OPHYS_2_DESCRIPTION),
    ("OPHYS_3_images_C", OPHYS_1_3_DESCRIPTION),
    ("OPHYS_4_images_D", OPHYS_4_6_DESCRIPTION),
    ("OPHYS_5_images_E", OPHYS_5_DESCRIPTION),
    ("OPHYS_6_images_F", OPHYS_4_6_DESCRIPTION)
])
def test_get_expt_description_with_valid_session_type(session_type,
                                                      expected_description):
    obt = get_expt_description(session_type)
    assert obt == expected_description


@pytest.mark.parametrize("session_type", [
    ("bogus_session_type"),
    ("stuff"),
    ("OPHYS_7")
])
def test_get_expt_description_raises_with_invalid_session_type(session_type):
    error_msg_match_phrase = r"Encountered an unknown session type*"
    with pytest.raises(RuntimeError, match=error_msg_match_phrase):
        _ = get_expt_description(session_type)
