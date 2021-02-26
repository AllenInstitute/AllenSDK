from datetime import datetime
import pickle
import pytest
import pytz

from allensdk.brain_observatory.behavior.session_apis\
    .data_transforms.behavior_data_transforms import BehaviorDataTransforms


@pytest.mark.parametrize("test_params, expected_warn_msg", [
    # Vanilla test case
    ({
        "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                 "%Y-%m-%d %H:%M:%S"),
        "pkl_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                           "%Y-%m-%d %H:%M:%S"),
        "behavior_session_id": 1
     },
     None
     ),

    # pkl expt date stored in unix format
    ({
        "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                 "%Y-%m-%d %H:%M:%S"),
        "pkl_expt_date": 1615716855.0,
        "behavior_session_id": 2
     },
     None
     ),

    # Extractor and pkl dates differ significantly
    ({
        "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                 "%Y-%m-%d %H:%M:%S"),
        "pkl_expt_date": datetime.strptime("2021-03-14 20:14:15",
                                           "%Y-%m-%d %H:%M:%S"),
        "behavior_session_id": 3
     },
     "The `date_of_acquisition` field in LIMS *"
     ),

    # pkl file contains an unparseable datetime
    ({
        "extractor_expt_date": datetime.strptime("2021-03-14 03:14:15",
                                                 "%Y-%m-%d %H:%M:%S"),
        "pkl_expt_date": None,
        "behavior_session_id": 4
     },
     "Could not parse the acquisition datetime *"
     ),
])
def test_get_experiment_date(tmp_path, test_params, expected_warn_msg):

    mock_session_id = test_params["behavior_session_id"]

    pkl_save_path = tmp_path / f"mock_pkl_{mock_session_id}.pkl"
    with open(pkl_save_path, 'wb') as handle:
        pickle.dump({"start_time": test_params['pkl_expt_date']}, handle)

    tz = pytz.timezone("America/Los_Angeles")
    extractor_expt_date = tz.localize(
        test_params['extractor_expt_date']).astimezone(pytz.utc)

    class MockExtractor():
        def get_experiment_date(self):
            return extractor_expt_date

        def get_behavior_session_id(self):
            return test_params['behavior_session_id']

        def get_behavior_stimulus_file(self):
            return pkl_save_path

    mock_extractor = MockExtractor()
    transformer_instance = BehaviorDataTransforms(extractor=mock_extractor)

    if expected_warn_msg:
        with pytest.warns(Warning, match=expected_warn_msg):
            obt_date = transformer_instance.get_experiment_date()
    else:
        obt_date = transformer_instance.get_experiment_date()

    assert(obt_date == extractor_expt_date)
