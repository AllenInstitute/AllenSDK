import pytest
import numpy as np

from allensdk.brain_observatory.behavior.metadata_processing import (
    description_dict, get_task_parameters, get_expt_description)


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
                    "flash_omit_probability": 0.05
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
        "omitted_flash_fraction": 0.05,
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
    ("OPHYS_0_images_Z", description_dict[r"\AOPHYS_0_images"]),
    ("OPHYS_1_images_A", description_dict[r"\AOPHYS_[1|3]_images"]),
    ("OPHYS_2_images_B", description_dict[r"\AOPHYS_2_images"]),
    ("OPHYS_3_images_C", description_dict[r"\AOPHYS_[1|3]_images"]),
    ("OPHYS_4_images_D", description_dict[r"\AOPHYS_[4|6]_images"]),
    ("OPHYS_5_images_E", description_dict[r"\AOPHYS_5_images"]),
    ("OPHYS_6_images_F", description_dict[r"\AOPHYS_[4|6]_images"]),
    ("TRAINING_0_gratings_A", description_dict[r"\ATRAINING_0_gratings"]),
    ("TRAINING_1_gratings_B", description_dict[r"\ATRAINING_1_gratings"]),
    ("TRAINING_2_gratings_C", description_dict[r"\ATRAINING_2_gratings"]),
    ("TRAINING_3_images_D", description_dict[r"\ATRAINING_3_images"]),
    ("TRAINING_4_images_E", description_dict[r"\ATRAINING_4_images"]),
    ('TRAINING_3_images_A_10uL_reward',
     description_dict[r"\ATRAINING_3_images"]),
    ('TRAINING_5_images_A_handoff_lapsed',
     description_dict[r"\ATRAINING_5_images"])
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
    with pytest.raises(RuntimeError, match=f"session type should match.*"):
        get_expt_description(session_type)
