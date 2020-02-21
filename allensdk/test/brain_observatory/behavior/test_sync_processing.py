import os
import pytest
import numpy as np
import allensdk.brain_observatory.behavior.sync as sync
from allensdk.brain_observatory.sync_dataset import Dataset

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
    obt = sync.get_sync_data(sync_path)[sync_key]
    assert count_exp == len(obt)
    assert last_exp == obt[-1]


@pytest.mark.parametrize("fn, key, rise, fall, expect", [
    [sync.get_trigger, "foo", None, None, None],
    [sync.get_trigger, "2p_trigger", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_trigger, "acq_trigger", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_eye_tracking, "cam2_exposure", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_eye_tracking, "eye_tracking", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_behavior_monitoring, "cam1_exposure", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_behavior_monitoring, "behavior_monitoring", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_stim_photodiode, "stim_photodiode", [1, 2, 3], [4, 5, 6], [1, 2, 3, 4, 5, 6]],
    [sync.get_stim_photodiode, "photodiode", [1, 2, 3], [4, 5, 6], [1, 2, 3, 4, 5, 6]],
    [sync.get_lick_times, "lick_times", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_lick_times, "lick_sensor", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_ophys_frames, "2p_vsync", [1, 2, 3], [4, 5, 6], [1, 2, 3]],
    [sync.get_raw_stimulus_frames, "stim_vsync", [1, 2, 3], [4, 5, 6], [4, 5, 6]],
])
def test_timestamp_extractors(fn, key, rise, fall, expect):

    class Ds(Dataset):
        def __init__(self):
            self.line_labels = [key, "1", "2"]

        def get_rising_edges(self, line, units):
            if not line in self.line_labels:
                raise ValueError
            return rise

        def get_falling_edges(self, line, units):
            if not line in self.line_labels:
                raise ValueError
            return fall

    if expect is None:
        with pytest.raises(KeyError) as _err:
            fn(Ds())
    else:
        assert np.allclose(expect, fn(Ds()))

