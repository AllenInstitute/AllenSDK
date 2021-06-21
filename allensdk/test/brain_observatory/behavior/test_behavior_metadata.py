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


def test_cre_line():
    """Tests that cre_line properly parsed from driver_line"""
    fg = FullGenotype(
        full_genotype='Sst-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt')
    assert fg.parse_cre_line() == 'Sst-IRES-Cre'


def test_cre_line_bad_full_genotype():
    """Test that cre_line is None and no error raised"""
    fg = FullGenotype(full_genotype='foo')

    with pytest.warns(UserWarning) as record:
        cre_line = fg.parse_cre_line(warn=True)
    assert cre_line is None
    assert str(record[0].message) == 'Unable to parse cre_line from ' \
                                     'full_genotype'


def test_reporter_line():
    """Test that reporter line properly parsed from list"""
    reporter_line = ReporterLine._parse(reporter_line=['foo'])
    assert reporter_line == 'foo'


def test_reporter_line_str():
    """Test that reporter line returns itself if str"""
    reporter_line = ReporterLine._parse(reporter_line='foo')
    assert reporter_line == 'foo'


@pytest.mark.parametrize("input_reporter_line, warning_msg, expected", (
        (('foo', 'bar'), 'More than 1 reporter line. '
                         'Returning the first one', 'foo'),
        (None, 'Error parsing reporter line. It is null.', None),
        ([], 'Error parsing reporter line. The array is empty', None)
)
                         )
def test_reporter_edge_cases(input_reporter_line, warning_msg,
                             expected):
    """Test reporter line edge cases"""
    with pytest.warns(UserWarning) as record:
        reporter_line = ReporterLine._parse(reporter_line=input_reporter_line,
                                            warn=True)
    assert reporter_line == expected
    assert str(record[0].message) == warning_msg


def test_age_in_days():
    """Test that age_in_days properly parsed from age"""
    age = Age._age_code_to_days(age='P123')
    assert age == 123


@pytest.mark.parametrize("input_age, warning_msg, expected", (
        ('unkown', 'Could not parse numeric age from age code '
                   '(age code does not start with "P")', None),
        ('P', 'Could not parse numeric age from age code '
              '(no numeric values found in age code)', None)
)
                         )
def test_age_in_days_edge_cases(monkeypatch, input_age, warning_msg,
                                expected):
    """Test age in days edge cases"""
    with pytest.warns(UserWarning) as record:
        age_in_days = Age._age_code_to_days(age=input_age, warn=True)

    assert age_in_days is None
    assert str(record[0].message) == warning_msg


@pytest.mark.parametrize("test_params, expected_warn_msg", [
    # Vanilla test case
    ({
         "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                  "%Y-%m-%d %H:%M:%S"),
         "pkl_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                            "%Y-%m-%d %H:%M:%S"),
         "behavior_session_id": 1
     }, None),

    # pkl expt date stored in unix format
    ({
         "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                  "%Y-%m-%d %H:%M:%S"),
         "pkl_expt_date": 1615716855.0,
         "behavior_session_id": 2
     }, None),

    # Extractor and pkl dates differ significantly
    ({
         "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                  "%Y-%m-%d %H:%M:%S"),
         "pkl_expt_date": datetime.strptime("2021-03-14 20:14:15",
                                            "%Y-%m-%d %H:%M:%S"),
         "behavior_session_id": 3
     },
     "The `date_of_acquisition` field in LIMS *"),

    # pkl file contains an unparseable datetime
    ({
         "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                  "%Y-%m-%d %H:%M:%S"),
         "pkl_expt_date": None,
         "behavior_session_id": 4
     },
     "Could not parse the acquisition datetime *"),
])
def test_get_date_of_acquisition(tmp_path, test_params,
                                 expected_warn_msg):
    mock_session_id = test_params["behavior_session_id"]

    pkl_save_path = tmp_path / f"mock_pkl_{mock_session_id}.pkl"
    with open(pkl_save_path, 'wb') as handle:
        pickle.dump({"start_time": test_params['pkl_expt_date']}, handle)

    tz = pytz.timezone("America/Los_Angeles")
    extractor_expt_date = tz.localize(
        test_params['extractor_expt_date']).astimezone(pytz.utc)

    stimulus_file = StimulusFile(filepath=pkl_save_path)
    obt_date = DateOfAcquisition(
        date_of_acquisition=extractor_expt_date)

    if expected_warn_msg:
        with pytest.warns(Warning, match=expected_warn_msg):
            obt_date.validate(
                stimulus_file=stimulus_file,
                behavior_session_id=test_params['behavior_session_id'])

    assert obt_date.value == extractor_expt_date


def test_indicator():
    """Test that indicator is parsed from full_genotype"""
    reporter_line = ReporterLine(reporter_line='Ai148(TIT2L-GC6f-ICL-tTA2)')
    assert reporter_line.parse_indicator() == 'GCaMP6f'


@pytest.mark.parametrize("input_reporter_line, warning_msg, expected", (
        (None, 'Could not parse indicator from reporter because there is no '
               'reporter', None),
        ('foo', 'Could not parse indicator from reporter because none'
                'of the expected substrings were found in the reporter', None)
)
                         )
def test_indicator_edge_cases(input_reporter_line, warning_msg,
                              expected):
    """Test indicator parsing edge cases"""
    with pytest.warns(UserWarning) as record:
        reporter_line = ReporterLine(reporter_line=input_reporter_line)
        indicator = reporter_line.parse_indicator(warn=True)
    assert indicator is expected
    assert str(record[0].message) == warning_msg
