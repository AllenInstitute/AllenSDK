from allensdk.brain_observatory.behavior.\
    data_objects.eye_tracking.eye_tracking_table import EyeTrackingTable


def test_incomplete_eye_tracking(
        behavior_ophys_experiment_fixture,
        skeletal_nwb_fixture):

    populated_eye_tracking = behavior_ophys_experiment_fixture.eye_tracking
    empty_eye_tracking = EyeTrackingTable.from_nwb(skeletal_nwb_fixture).value

    assert len(populated_eye_tracking) > 0
    assert len(empty_eye_tracking) == 0

    populated_columns = set(populated_eye_tracking.columns)
    empty_columns = set(empty_eye_tracking.columns)
    assert populated_columns == empty_columns

    assert populated_eye_tracking.index.name == empty_eye_tracking.index.name
