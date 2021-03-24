import math
from datetime import datetime
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
import pytz

from allensdk import OneResultExpectedError
from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.mtrain import ExtendedTrialSchema
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorLimsApi, BehaviorLimsExtractor, BehaviorOphysLimsApi)
from allensdk.core.authentication import DbCredentials
from allensdk.core.exceptions import DataFrameIndexError
from marshmallow.schema import ValidationError


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_experiment_id, compare_val', [
    pytest.param(880293569,
                 ("/allen/programs/braintv/production/neuralcoding/prod0"
                  "/specimen_703198163/behavior_session_880293569/"
                  "880289456.pkl")),
    pytest.param(0, None)
])
def test_get_behavior_stimulus_file(behavior_experiment_id, compare_val):

    if compare_val is None:
        expected_fail = False
        try:
            api = BehaviorLimsApi(behavior_experiment_id)
            api.extractor.get_behavior_stimulus_file()
        except OneResultExpectedError:
            expected_fail = True
        assert expected_fail is True
    else:
        api = BehaviorLimsApi(behavior_experiment_id)
        assert api.extractor.get_behavior_stimulus_file() == compare_val


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_session_uuid', [
    pytest.param('394a910e-94c7-4472-9838-5345aff59ed8'),
])
def test_foraging_id_to_behavior_session_id(behavior_session_uuid):
    session = BehaviorLimsExtractor.from_foraging_id(behavior_session_uuid)
    assert session.behavior_session_id == 823847007


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('behavior_session_id', [
    pytest.param(823847007),
])
def test_behavior_session_id_to_foraging_id(behavior_session_id):
    session = BehaviorLimsApi(behavior_session_id=behavior_session_id)
    behavior_session_uuid = session.get_metadata().behavior_session_uuid
    expected = UUID('394a910e-94c7-4472-9838-5345aff59ed8')
    assert behavior_session_uuid == expected


@pytest.mark.requires_bamboo
@pytest.mark.parametrize(
    'behavior_experiment_id', [
        880293569,  # stage: TRAINING_0_gratings_autorewards_15min
        881236782,  # stage: TRAINING_1_gratings
        881236761,  # stage: TRAINING_2_gratings_flashed
    ]
)
def test_get_extended_trials(behavior_experiment_id):
    api = BehaviorLimsApi(behavior_experiment_id)
    df = api.get_extended_trials()
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df.to_dict('records')
    data_list_cs_sc = ets.dump(data_list_cs)
    ets.load(data_list_cs_sc)

    df_fail = df.drop(['behavior_session_uuid'], axis=1)
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df_fail.to_dict('records')
    data_list_cs_sc = ets.dump(data_list_cs)
    try:
        ets.load(data_list_cs_sc)
        raise RuntimeError("This should have failed with "
                           "marshmallow.schema.ValidationError")
    except ValidationError:
        pass


mock_db_credentials = DbCredentials(dbname='mock_db', user='mock_user',
                                    host='mock_host', port='mock_port',
                                    password='mock')


@pytest.fixture
def MockBehaviorLimsApi():

    class MockBehaviorLimsRawApi(BehaviorLimsExtractor):
        """
        Mock class that overrides some functions to provide test data and
        initialize without calls to db.
        """
        def __init__(self):
            super().__init__(behavior_session_id=8675309,
                             lims_credentials=mock_db_credentials,
                             mtrain_credentials=mock_db_credentials)
            self.foraging_id = '138531ab-fe59-4523-9154-07c8d97bbe03'

        def _get_ids(self):
            return {}

        def get_date_of_acquisition(self):
            return datetime(2019, 9, 26, 16, tzinfo=pytz.UTC)

        def get_behavior_stimulus_file(self):
            return "dummy_stimulus_file.pkl"

        def get_foraging_id(self) -> str:
            return self.foraging_id

    class MockBehaviorLimsApi(BehaviorLimsApi):

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
                "session_uuid": '138531ab-fe59-4523-9154-07c8d97bbe03',
                "start_time": datetime(2019, 9, 26, 9),
            }
            return data

        def get_running_acquisition_df(self, lowpass=True):
            return pd.DataFrame(
                {"timestamps": [0.0, 0.1, 0.2],
                 "speed": [8.0, 15.0, 16.0]}).set_index("timestamps")

    api = MockBehaviorLimsApi(extractor=MockBehaviorLimsRawApi())
    yield api
    api.cache_clear()


@pytest.fixture
def MockApiRunSpeedExpectedError():
    class MockApiRunSpeedExpectedError(BehaviorLimsExtractor):
        """
        Mock class that overrides some functions to provide test data and
        initialize without calls to db.
        """
        def __init__(self):
            super().__init__(behavior_session_id=8675309,
                             mtrain_credentials=mock_db_credentials,
                             lims_credentials=mock_db_credentials)

        def _get_ids(self):
            return {}

    class MockBehaviorLimsApiRunSpeedExpectedError(BehaviorLimsApi):

        def get_running_acquisition_df(self, lowpass=True):
            return pd.DataFrame(
                {"timestamps": [0.0, 0.1, 0.2],
                 "speed": [8.0, 15.0, 16.0]})

    return MockBehaviorLimsApiRunSpeedExpectedError(
        extractor=MockApiRunSpeedExpectedError())


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
def test_get_stimulus_timestamps(MockBehaviorLimsApi):
    api = MockBehaviorLimsApi
    expected = np.array([0.016 * i for i in range(11)])
    assert np.allclose(expected, api.get_stimulus_timestamps())


def test_get_licks(MockBehaviorLimsApi):
    api = MockBehaviorLimsApi
    expected = pd.DataFrame({"timestamps": [0.016 * i for i in [2., 6., 9.]],
                             "frame": [2, 6, 9]})
    pd.testing.assert_frame_equal(expected, api.get_licks())


def test_get_behavior_session_uuid(MockBehaviorLimsApi, monkeypatch):
    with monkeypatch.context() as ctx:
        def dummy_init(self, extractor, behavior_stimulus_file):
            self._extractor = extractor
            self._behavior_stimulus_file = behavior_stimulus_file

        ctx.setattr(BehaviorMetadata,
                    '__init__',
                    dummy_init)
        stimulus_file = MockBehaviorLimsApi._behavior_stimulus_file()
        metadata = BehaviorMetadata(
            extractor=MockBehaviorLimsApi.extractor,
            behavior_stimulus_file=stimulus_file)

    expected = UUID('138531ab-fe59-4523-9154-07c8d97bbe03')
    assert expected == metadata.behavior_session_uuid


def test_get_stimulus_frame_rate(MockBehaviorLimsApi):
    api = MockBehaviorLimsApi
    assert 62.0 == api.get_stimulus_frame_rate()


def test_get_date_of_acquisition(MockBehaviorLimsApi):
    api = MockBehaviorLimsApi
    expected = datetime(2019, 9, 26, 16, tzinfo=pytz.UTC)
    actual = api.get_metadata().date_of_acquisition
    assert expected == actual


def test_get_running_speed(MockBehaviorLimsApi):
    expected = pd.DataFrame({
        "timestamps": [0.0, 0.1, 0.2],
        "speed": [8.0, 15.0, 16.0]})
    api = MockBehaviorLimsApi
    actual = api.get_running_speed()
    pd.testing.assert_frame_equal(expected, actual)


def test_get_running_speed_raises_index_error(MockApiRunSpeedExpectedError):
    with pytest.raises(DataFrameIndexError):
        MockApiRunSpeedExpectedError.get_running_speed()


# def test_get_stimulus_presentations(MockBehaviorLimsApi):
#     api = MockBehaviorLimsApi
#     #  TODO. This function is a monster with multiple dependencies,
#     #  no tests, and no documentation (for any of its dependencies).
#     #  Needs to be broken out into testable parts.

@pytest.mark.requires_bamboo
@pytest.mark.nightly
class TestBehaviorRegression:
    """
    Test whether behavior sessions (that are also ophys) loaded with
    BehaviorLimsExtractor return the same results as sessions loaded
    with BehaviorOphysLimsExtractor, for relevant functions. Do not check for
    timestamps, which are from different files so will not be the same.
    Also not checking for experiment_date, since they're from two different
    sources (and I'm not sure how it's uploaded in the database).

    Do not test `get_licks` regression because the licks come from two
    different sources and are recorded differently (behavior pickle file in
    BehaviorLimsExtractor; sync file in BehaviorOphysLimeApi)
    """
    @classmethod
    def setup_class(cls):
        cls.bd = BehaviorLimsApi(976012750)
        cls.od = BehaviorOphysLimsApi(976255949)

    @classmethod
    def teardown_class(cls):
        cls.bd.cache_clear()
        cls.od.cache_clear()

    def test_stim_file_regression(self):
        assert (self.bd.extractor.get_behavior_stimulus_file()
                == self.od.extractor.get_behavior_stimulus_file())

    def test_get_rewards_regression(self):
        bd_rewards = self.bd.get_rewards().drop(columns=['timestamps'])
        od_rewards = self.od.get_rewards().drop(columns=['timestamps'])
        pd.testing.assert_frame_equal(bd_rewards, od_rewards)

    def test_ophys_experiment_id_regression(self):
        assert (self.bd.extractor.ophys_experiment_ids[0]
                == self.od.extractor.ophys_experiment_id)

    def test_behavior_uuid_regression(self):
        assert (self.bd.get_metadata().behavior_session_uuid
                == self.od.get_metadata().behavior_session_uuid)

    def test_container_id_regression(self):
        assert (self.bd.extractor.ophys_container_id
                == self.od.extractor.get_ophys_container_id())

    def test_stimulus_frame_rate_regression(self):
        assert (self.bd.get_stimulus_frame_rate()
                == self.od.get_stimulus_frame_rate())

    def test_get_running_speed_regression(self):
        """Can't test values because they're intrinsically linked to timestamps
        """
        bd_speed = self.bd.get_running_speed(lowpass=False)
        od_speed = self.od.get_running_speed(lowpass=False)

        assert len(bd_speed.values) == len(od_speed.values)
        assert len(bd_speed.timestamps) == len(od_speed.timestamps)

    def test_get_running_acquisition_df_regression(self):
        """Can't test values because they're intrinsically linked to timestamps
        """
        bd_running = self.bd.get_running_acquisition_df(lowpass=False)
        od_running = self.od.get_running_acquisition_df(lowpass=False)

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
            bd_template_img = bd_template[k]
            od_template_img = od_template[k]

            assert np.allclose(bd_template_img.unwarped,
                               od_template_img.unwarped,
                               equal_nan=True)
            assert np.allclose(bd_template_img.warped,
                               od_template_img.warped,
                               equal_nan=True)

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
        assert self.bd.extractor.get_sex() == self.od.extractor.get_sex()

    def test_get_equipment_name_regression(self):
        assert (self.bd.extractor.get_equipment_name()
                == self.od.extractor.get_equipment_name())

    def test_get_stimulus_name_regression(self):
        assert (self.bd.extractor.get_stimulus_name()
                == self.od.extractor.get_stimulus_name())

    def test_get_reporter_line_regression(self):
        assert (self.bd.extractor.get_reporter_line()
                == self.od.extractor.get_reporter_line())

    def test_get_driver_line_regression(self):
        assert (self.bd.extractor.get_driver_line()
                == self.od.extractor.get_driver_line())

    def test_get_external_specimen_name_regression(self):
        assert (self.bd.extractor.get_mouse_id()
                == self.od.extractor.get_mouse_id())

    def test_get_full_genotype_regression(self):
        assert (self.bd.extractor.get_full_genotype()
                == self.od.extractor.get_full_genotype())

    def test_get_date_of_acquisition_regression(self):
        """Just testing the date since it comes from two different sources;
        We expect that BehaviorOphysLimsApi will be earlier (more like when
        rig was started up), while BehaviorLimsExtractor returns the start of
        the actual behavior (from pkl file)"""
        assert (self.bd.get_metadata().date_of_acquisition.date()
                == self.od.get_metadata().date_of_acquisition.date())
