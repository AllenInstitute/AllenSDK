import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import math

from allensdk.internal.api.behavior_data_lims_api import BehaviorDataLimsApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.running_speed import RunningSpeed
from allensdk.core.exceptions import DataFrameIndexError


@pytest.fixture
def MockBehaviorDataLimsApi():
    class MockBehaviorDataLimsApi(BehaviorDataLimsApi):
        """
        Mock class that overrides some functions to provide test data and
        initialize without calls to db.
        """
        def __init__(self):
            super().__init__(behavior_session_id=8675309)

        def _get_ids(self):
            return {}

        def _behavior_stimulus_file(self):
            data = {
                "items": {
                    "behavior": {
                        "lick_sensors": [{
                            "lick_events": [2, 6, 9],
                        }],
                        "intervalsms": np.array([16.0]*10),
                        "trial_log": [
                            {
                                "events":
                                    [
                                        ["trial_start", 2, 2],
                                        ["trial_end", 3, 3],
                                    ],
                            },
                            {
                                "events":
                                    [
                                        ["trial_start", 4, 4],
                                        ["trial_start", 5, 5],
                                    ],
                            },
                        ],
                    },
                },
                "session_uuid": 123456,
                "start_time": datetime(2019, 9, 26, 9),
            }
            return data

        def get_running_data_df(self):
            return pd.DataFrame(
                {"timestamps": [0.0, 0.1, 0.2],
                 "speed": [8.0, 15.0, 16.0]}).set_index("timestamps")

    api = MockBehaviorDataLimsApi()
    yield api
    api.cache_clear()


@pytest.fixture
def MockApiRunSpeedExpectedError():
    class MockApiRunSpeedExpectedError(BehaviorDataLimsApi):
        """
        Mock class that overrides some functions to provide test data and 
        initialize without calls to db.
        """
        def __init__(self):
            super().__init__(behavior_session_id=8675309)

        def _get_ids(self):
            return {}

        def get_running_data_df(self):
            return pd.DataFrame(
                {"timestamps": [0.0, 0.1, 0.2],
                 "speed": [8.0, 15.0, 16.0]})
    return MockApiRunSpeedExpectedError()


# Test the non-sql-query functions
# Does not include tests for the following functions, as they are just calls to
# static methods provided for convenience (and should be covered with their own
# unit tests):
#    get_rewards
#    get_running_data_df
#    get_stimulus_templates
#    get_task_parameters
#    get_trials
# Does not include test for get_metadata since it just collects data from
# methods covered in other unit tests, or data derived from sql queries.
def test_get_stimulus_timestamps(MockBehaviorDataLimsApi):
    api = MockBehaviorDataLimsApi
    expected = np.array([0.016 * i for i in range(11)])
    assert np.allclose(expected, api.get_stimulus_timestamps())


def test_get_licks(MockBehaviorDataLimsApi):
    api = MockBehaviorDataLimsApi
    expected = pd.DataFrame({"time": [0.016 * i for i in [2., 6., 9.]]})
    pd.testing.assert_frame_equal(expected, api.get_licks())


def test_get_behavior_session_uuid(MockBehaviorDataLimsApi):
    api = MockBehaviorDataLimsApi
    assert 123456 == api.get_behavior_session_uuid()


def test_get_stimulus_frame_rate(MockBehaviorDataLimsApi):
    api = MockBehaviorDataLimsApi
    assert 62.0 == api.get_stimulus_frame_rate()


def test_get_experiment_date(MockBehaviorDataLimsApi):
    api = MockBehaviorDataLimsApi
    expected = datetime(2019, 9, 26, 16, tzinfo=pytz.UTC)
    actual = api.get_experiment_date()
    assert expected == actual


def test_get_running_speed(MockBehaviorDataLimsApi):
    expected = RunningSpeed(timestamps=[0.0, 0.1, 0.2],
                            values=[8.0, 15.0, 16.0])
    api = MockBehaviorDataLimsApi
    actual = api.get_running_speed()
    assert expected == actual


def test_get_running_speed_raises_index_error(MockApiRunSpeedExpectedError):
    with pytest.raises(DataFrameIndexError):
        MockApiRunSpeedExpectedError.get_running_speed()


# def test_get_stimulus_presentations(MockBehaviorDataLimsApi):
#     api = MockBehaviorDataLimsApi
#     #  TODO. This function is a monster with multiple dependencies,
#     #  no tests, and no documentation (for any of its dependencies).
#     #  Needs to be broken out into testable parts.

@pytest.mark.requires_bamboo
@pytest.mark.nightly
class TestBehaviorRegression:
    """
    Test whether behavior sessions (that are also ophys) loaded with
    BehaviorDataLimsApi return the same results as sessions loaded
    with BehaviorOphysLimsApi, for relevant functions. Do not check for
    timestamps, which are from different files so will not be the same.
    Also not checking for experiment_date, since they're from two different
    sources (and I'm not sure how it's uploaded in the database).

    Do not test `get_licks` regression because the licks come from two
    different sources and are recorded differently (behavior pickle file in 
    BehaviorDataLimsApi; sync file in BehaviorOphysLimeApi)
    """
    @classmethod
    def setup_class(cls):
        cls.bd = BehaviorDataLimsApi(976012750)
        cls.od = BehaviorOphysLimsApi(976255949)

    @classmethod
    def teardown_class(cls):
        cls.bd.cache_clear()
        cls.od.cache_clear()

    def test_stim_file_regression(self):
        assert (self.bd.get_behavior_stimulus_file()
                == self.od.get_behavior_stimulus_file())

    def test_get_rewards_regression(self):
        """Index is timestamps here, so remove it before comparing."""
        bd_rewards = self.bd.get_rewards().reset_index(drop=True)
        od_rewards = self.od.get_rewards().reset_index(drop=True)
        pd.testing.assert_frame_equal(bd_rewards, od_rewards)

    def test_ophys_experiment_id_regression(self):
        assert self.bd.ophys_experiment_ids[0] == self.od.ophys_experiment_id

    def test_behavior_uuid_regression(self):
        assert (self.bd.get_behavior_session_uuid()
                == self.od.get_behavior_session_uuid())

    def test_container_id_regression(self):
        assert (self.bd.ophys_container_id
                == self.od.get_experiment_container_id())

    def test_stimulus_frame_rate_regression(self):
        assert (self.bd.get_stimulus_frame_rate()
                == self.od.get_stimulus_frame_rate())

    def test_get_running_speed_regression(self):
        """Can't test values because they're intrinsically linked to timestamps
        """
        bd_speed = self.bd.get_running_speed()
        od_speed = self.od.get_running_speed()
        assert len(bd_speed.values) == len(od_speed.values)
        assert len(bd_speed.timestamps) == len(od_speed.timestamps)

    def test_get_running_df_regression(self):
        """Can't test values because they're intrinsically linked to timestamps
        """
        bd_running = self.bd.get_running_data_df()
        od_running = self.od.get_running_data_df()
        assert len(bd_running) == len(od_running)
        assert list(bd_running) == list(od_running)

    def test_get_stimulus_presentations_regression(self):
        drop_cols = ["start_time", "stop_time"]
        bd_pres = self.bd.get_stimulus_presentations().drop(drop_cols, axis=1)
        od_pres = self.od.get_stimulus_presentations().drop(drop_cols, axis=1)
        # Duration needs less precision (timestamp-dependent)
        pd.testing.assert_frame_equal(bd_pres, od_pres, check_less_precise=2)

    def test_get_stimulus_template_regression(self):
        bd_template = self.bd.get_stimulus_templates()
        od_template = self.od.get_stimulus_templates()
        assert bd_template.keys() == od_template.keys()
        for k in bd_template.keys():
            assert np.array_equal(bd_template[k], od_template[k])

    def test_get_task_parameters_regression(self):
        bd_params = self.bd.get_task_parameters()
        od_params = self.od.get_task_parameters()
        # Have to do special checking because of nan equality
        assert bd_params.keys() == od_params.keys()
        for k in bd_params.keys():
            bd_v = bd_params[k]
            od_v = od_params[k]
            try:
                if math.isnan(bd_v):
                    assert math.isnan(od_v)
                else:
                    assert bd_v == od_v
            except (AttributeError, TypeError):
                assert bd_v == od_v

    def test_get_trials_regression(self):
        """ A lot of timestamp dependent values. Test what we can."""
        cols_to_test = ["reward_volume", "hit", "false_alarm", "miss",
                        "stimulus_change", "aborted", "go",
                        "catch", "auto_rewarded", "correct_reject",
                        "trial_length", "change_frame", "initial_image_name",
                        "change_image_name"]
        bd_trials = self.bd.get_trials()[cols_to_test]
        od_trials = self.od.get_trials()[cols_to_test]
        pd.testing.assert_frame_equal(bd_trials, od_trials,
                                      check_less_precise=2)

    def test_get_sex_regression(self):
        assert self.bd.get_sex() == self.od.get_sex()

    def test_get_rig_name_regression(self):
        assert self.bd.get_rig_name() == self.od.get_rig_name()

    def test_get_stimulus_name_regression(self):
        assert self.bd.get_stimulus_name() == self.od.get_stimulus_name()

    def test_get_reporter_line_regression(self):
        assert self.bd.get_reporter_line() == self.od.get_reporter_line()

    def test_get_driver_line_regression(self):
        assert self.bd.get_driver_line() == self.od.get_driver_line()

    def test_get_external_specimen_name_regression(self):
        assert (self.bd.get_external_specimen_name()
                == self.od.get_external_specimen_name())

    def test_get_full_genotype_regression(self):
        assert self.bd.get_full_genotype() == self.od.get_full_genotype()

    def test_get_experiment_date_regression(self):
        """Just testing the date since it comes from two different sources;
        We expect that BehaviorOphysLimsApi will be earlier (more like when
        rig was started up), while BehaviorDataLimsApi returns the start of
        the actual behavior (from pkl file)"""
        assert (self.bd.get_experiment_date().date()
                == self.od.get_experiment_date().date())
