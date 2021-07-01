import pickle
from datetime import datetime

import pytest
import numpy as np
import pandas as pd
import pytz

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import (
    description_dict, get_task_parameters, get_expt_description,
    BehaviorMetadata)
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.age import \
    Age
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.full_genotype import \
    FullGenotype
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.reporter_line import \
    ReporterLine


@pytest.mark.parametrize("data, expected",
                         [pytest.param({  # noqa: E128
                             "items": {
                                 "behavior": {
                                     "config": {
                                         "DoC": {
                                             "blank_duration_range": (
                                                     0.5, 0.6),
                                             "response_window": [0.15, 0.75],
                                             "change_time_dist": "geometric",
                                             "auto_reward_volume": 0.002,
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
                                         "images": {"draw_log": [1] * 10,
                                                    "flash_interval_sec": [
                                                        0.32, -1.0]}
                                     },
                                 }
                             }
                         },
                             {
                                 "blank_duration_sec": [0.5, 0.6],
                                 "stimulus_duration_sec": 0.32,
                                 "omitted_flash_fraction": 0.05,
                                 "response_window_sec": [0.15, 0.75],
                                 "reward_volume": 0.007,
                                 "session_type": "TRAINING_3_images_A",
                                 "stimulus": "images",
                                 "stimulus_distribution": "geometric",
                                 "task": "change detection",
                                 "n_stimulus_frames": 10,
                                 "auto_reward_volume": 0.002
                             }, id='basic'),
                             pytest.param({
                                 "items": {
                                     "behavior": {
                                         "config": {
                                             "DoC": {
                                                 "blank_duration_range": (
                                                         0.5, 0.5),
                                                 "response_window": [0.15,
                                                                     0.75],
                                                 "change_time_dist":
                                                     "geometric",
                                                 "auto_reward_volume": 0.002
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
                                             "images": {"draw_log": [1] * 10,
                                                        "flash_interval_sec": [
                                                            0.32, -1.0]}
                                         },
                                     }
                                 }
                             },
                                 {
                                     "blank_duration_sec": [0.5, 0.5],
                                     "stimulus_duration_sec": 0.32,
                                     "omitted_flash_fraction": 0.05,
                                     "response_window_sec": [0.15, 0.75],
                                     "reward_volume": 0.007,
                                     "session_type": "TRAINING_3_images_A",
                                     "stimulus": "images",
                                     "stimulus_distribution": "geometric",
                                     "task": "change detection",
                                     "n_stimulus_frames": 10,
                                     "auto_reward_volume": 0.002
                                 }, id='single_value_blank_duration'),
                             pytest.param({
                                 "items": {
                                     "behavior": {
                                         "config": {
                                             "DoC": {
                                                 "blank_duration_range": (
                                                         0.5, 0.5),
                                                 "response_window": [0.15,
                                                                     0.75],
                                                 "change_time_dist":
                                                     "geometric",
                                                 "auto_reward_volume": 0.002
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
                                             "grating": {"draw_log": [1] * 10,
                                                         "flash_interval_sec":
                                                             [0.34, -1.0]}
                                         },
                                     }
                                 }
                             },
                                 {
                                     "blank_duration_sec": [0.5, 0.5],
                                     "stimulus_duration_sec": 0.34,
                                     "omitted_flash_fraction": 0.05,
                                     "response_window_sec": [0.15, 0.75],
                                     "reward_volume": 0.007,
                                     "session_type": "TRAINING_3_images_A",
                                     "stimulus": "grating",
                                     "stimulus_distribution": "geometric",
                                     "task": "change detection",
                                     "n_stimulus_frames": 10,
                                     "auto_reward_volume": 0.002
                                 }, id='stimulus_duration_from_grating'),
                             pytest.param({
                                 "items": {
                                     "behavior": {
                                         "config": {
                                             "DoC": {
                                                 "blank_duration_range": (
                                                         0.5, 0.5),
                                                 "response_window": [0.15,
                                                                     0.75],
                                                 "change_time_dist":
                                                     "geometric",
                                                 "auto_reward_volume": 0.002
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
                                             "grating": {
                                                 "draw_log": [1] * 10,
                                                 "flash_interval_sec": None}
                                         },
                                     }
                                 }
                             },
                                 {
                                     "blank_duration_sec": [0.5, 0.5],
                                     "stimulus_duration_sec": np.NaN,
                                     "omitted_flash_fraction": 0.05,
                                     "response_window_sec": [0.15, 0.75],
                                     "reward_volume": 0.007,
                                     "session_type": "TRAINING_3_images_A",
                                     "stimulus": "grating",
                                     "stimulus_distribution": "geometric",
                                     "task": "change detection",
                                     "n_stimulus_frames": 10,
                                     "auto_reward_volume": 0.002
                                 }, id='stimulus_duration_none')
                         ]
                         )
def test_get_task_parameters(data, expected):
    actual = get_task_parameters(data)
    for k, v in actual.items():
        # Special nan checking since pytest doesn't do it well
        try:
            if np.isnan(v):
                assert np.isnan(expected[k])
            else:
                assert expected[k] == v
        except (TypeError, ValueError):
            assert expected[k] == v

    actual_keys = list(actual.keys())
    actual_keys.sort()
    expected_keys = list(expected.keys())
    expected_keys.sort()
    assert actual_keys == expected_keys


def test_get_task_parameters_task_id_exception():
    """
    Test that, when task_id has an unexpected value,
    get_task_parameters throws the correct exception
    """
    input_data = {
        "items": {
            "behavior": {
                "config": {
                    "DoC": {
                        "blank_duration_range": (0.5, 0.6),
                        "response_window": [0.15, 0.75],
                        "change_time_dist": "geometric",
                        "auto_reward_volume": 0.002
                    },
                    "reward": {
                        "reward_volume": 0.007,
                    },
                    "behavior": {
                        "task_id": "junk",
                    },
                },
                "params": {
                    "stage": "TRAINING_3_images_A",
                    "flash_omit_probability": 0.05
                },
                "stimuli": {
                    "images": {"draw_log": [1] * 10,
                               "flash_interval_sec": [0.32, -1.0]}
                },
            }
        }
    }

    with pytest.raises(RuntimeError) as error:
        _ = get_task_parameters(input_data)
    assert "does not know how to parse 'task_id'" in error.value.args[0]


def test_get_task_parameters_flash_duration_exception():
    """
    Test that, when 'images' or 'grating' not present in 'stimuli',
    get_task_parameters throws the correct exception
    """
    input_data = {
        "items": {
            "behavior": {
                "config": {
                    "DoC": {
                        "blank_duration_range": (0.5, 0.6),
                        "response_window": [0.15, 0.75],
                        "change_time_dist": "geometric",
                        "auto_reward_volume": 0.002
                    },
                    "reward": {
                        "reward_volume": 0.007,
                    },
                    "behavior": {
                        "task_id": "DoC",
                    },
                },
                "params": {
                    "stage": "TRAINING_3_images_A",
                    "flash_omit_probability": 0.05
                },
                "stimuli": {
                    "junk": {"draw_log": [1] * 10,
                             "flash_interval_sec": [0.32, -1.0]}
                },
            }
        }
    }

    with pytest.raises(RuntimeError) as error:
        _ = get_task_parameters(input_data)
    shld_be = "'images' and/or 'grating' not a valid key"
    assert shld_be in error.value.args[0]


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
    with pytest.raises(RuntimeError, match="session type should match.*"):
        get_expt_description(session_type)
