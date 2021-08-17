import pickle
from datetime import datetime

import pytest
import numpy as np
import pandas as pd
import pytz

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import (
    description_dict, get_task_parameters, get_expt_description,
    BehaviorMetadata)


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
        get_task_parameters(input_data)
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
        get_task_parameters(input_data)
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


def test_cre_line(monkeypatch):
    """Tests that cre_line properly parsed from driver_line"""
    with monkeypatch.context() as ctx:
        def dummy_init(self):
            pass

        def full_genotype(self):
            return 'Sst-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt'

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)
        ctx.setattr(BehaviorMetadata,
                    'full_genotype',
                    property(full_genotype))

        metadata = BehaviorMetadata()

        assert metadata.cre_line == 'Sst-IRES-Cre'


def test_cre_line_bad_full_genotype(monkeypatch):
    """Test that cre_line is None and no error raised"""
    with monkeypatch.context() as ctx:
        def dummy_init(self):
            pass

        def full_genotype(self):
            return 'foo'

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)
        ctx.setattr(BehaviorMetadata,
                    'full_genotype',
                    property(full_genotype))

        metadata = BehaviorMetadata()

        with pytest.warns(UserWarning) as record:
            cre_line = metadata.cre_line
        assert cre_line is None
        assert str(record[0].message) == 'Unable to parse cre_line from ' \
                                         'full_genotype'


def test_reporter_line(monkeypatch):
    """Test that reporter line properly parsed from list"""

    class MockExtractor:
        def get_reporter_line(self):
            return ['foo']

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorMetadata()

        assert metadata.reporter_line == 'foo'


def test_reporter_line_str(monkeypatch):
    """Test that reporter line returns itself if str"""

    class MockExtractor:
        def get_reporter_line(self):
            return 'foo'

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorMetadata()

        assert metadata.reporter_line == 'foo'


@pytest.mark.parametrize("input_reporter_line, warning_msg, expected", (
        (('foo', 'bar'), 'More than 1 reporter line. '
                         'Returning the first one', 'foo'),
        (None, 'Error parsing reporter line. It is null.', None),
        ([], 'Error parsing reporter line. The array is empty', None)
)
                         )
def test_reporter_edge_cases(monkeypatch, input_reporter_line, warning_msg,
                             expected):
    """Test reporter line edge cases"""

    class MockExtractor:
        def get_reporter_line(self):
            return input_reporter_line

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)
        metadata = BehaviorMetadata()

        with pytest.warns(UserWarning) as record:
            reporter_line = metadata.reporter_line

        assert reporter_line == expected
        assert str(record[0].message) == warning_msg


def test_age_in_days(monkeypatch):
    """Test that age_in_days properly parsed from age"""

    class MockExtractor:
        def get_age(self):
            return 'P123'

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorMetadata()

        assert metadata.age_in_days == 123


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

    class MockExtractor:
        def get_age(self):
            return input_age

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorMetadata()

        with pytest.warns(UserWarning) as record:
            age_in_days = metadata.age_in_days

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
def test_get_date_of_acquisition(monkeypatch, tmp_path, test_params,
                                 expected_warn_msg):
    mock_session_id = test_params["behavior_session_id"]

    pkl_save_path = tmp_path / f"mock_pkl_{mock_session_id}.pkl"
    with open(pkl_save_path, 'wb') as handle:
        pickle.dump({"start_time": test_params['pkl_expt_date']}, handle)
    behavior_stimulus_file = pd.read_pickle(pkl_save_path)

    tz = pytz.timezone("America/Los_Angeles")
    extractor_expt_date = tz.localize(
        test_params['extractor_expt_date']).astimezone(pytz.utc)

    class MockExtractor():
        def get_date_of_acquisition(self):
            return extractor_expt_date

        def get_behavior_session_id(self):
            return test_params['behavior_session_id']

        def get_behavior_stimulus_file(self):
            return pkl_save_path

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self, extractor, behavior_stimulus_file):
            self._extractor = extractor
            self._behavior_stimulus_file = behavior_stimulus_file

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorMetadata(
            extractor=extractor,
            behavior_stimulus_file=behavior_stimulus_file)

        if expected_warn_msg:
            with pytest.warns(Warning, match=expected_warn_msg):
                obt_date = metadata.date_of_acquisition
        else:
            obt_date = metadata.date_of_acquisition

        assert obt_date == extractor_expt_date
