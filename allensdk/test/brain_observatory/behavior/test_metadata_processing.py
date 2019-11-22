import numpy as np

from allensdk.brain_observatory.behavior.metadata_processing import (
    get_task_parameters)


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
