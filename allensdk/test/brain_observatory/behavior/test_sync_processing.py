import os
import pytest
from allensdk.brain_observatory.behavior.sync import get_sync_data

base_dir = os.path.join(
    "/",
    "allen",
    "programs",
    "braintv",
    "production",
    "visualbehavior",
    "prod0",
    "specimen_789992909",
    "ophys_session_819949602",
)
sync_path=os.path.join(
    base_dir, 
    "819949602_sync.h5"
)


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("sync_path, sync_key, count_exp, last_exp", [
    [sync_path, "ophys_frames", 140082, 4530.11659],
    [sync_path, "lick_times", 2099, 3860.94482],
    [sync_path, "ophys_trigger", 1, 6.8612],
    [sync_path, "eye_tracking", 135908, 4531.00479],
    [sync_path, "behavior_monitoring", 135887, 4530.19092],
    [sync_path, "stim_photodiode", 4512, 4510.80997],
    [sync_path, "stimulus_times_no_delay", 269977, 4510.25654],
])
def test_get_time_sync_integration(sync_path, sync_key, count_exp, last_exp):
    obt = get_sync_data(sync_path)[sync_key]
    assert count_exp == len(obt)
    assert last_exp == obt[-1]


