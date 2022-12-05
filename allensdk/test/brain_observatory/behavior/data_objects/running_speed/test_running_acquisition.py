import pytest
from unittest.mock import create_autospec

import pandas as pd

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)
from allensdk.brain_observatory.behavior.data_objects import (
    RunningAcquisition, StimulusTimestamps
)
from allensdk.brain_observatory.behavior.data_files import (
    SyncFile)


def test_nonzero_monitor_delay_acq():
    """
    Test that RunningAcquisition throws an exception if instantiated
    with a timestamps object that has non-zero monitor_delay
    """
    class OtherTimestamps(object):
        monitor_delay = 0.01
        value = 0.0

    with pytest.raises(RuntimeError,
                       match="should be no monitor delay"):

        RunningAcquisition(
            running_acquisition=4.0,
            stimulus_file=None,
            stimulus_timestamps=OtherTimestamps())


@pytest.mark.skip('to_json not supported yet')
@pytest.mark.parametrize(
    "stimulus_file, stimulus_file_to_json_ret, "
    "stimulus_timestamps, stimulus_timestamps_to_json_ret, raises, expected",
    [
        # Test to_json with both stimulus_file and sync_file
        (
            # stimulus_file
            create_autospec(BehaviorStimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # stimulus_timestamps
            create_autospec(StimulusTimestamps, instance=True),
            # stimulus_timestamps_to_json_ret
            {"sync_file": "sync.h5"},
            # raises
            False,
            # expected
            {"behavior_stimulus_file": "stim.pkl", "sync_file": "sync.h5"}
        ),
        # Test to_json without stimulus_file
        (
            # stimulus_file
            None,
            # stimulus_file_to_json_ret
            None,
            # stimulus_timestamps
            create_autospec(StimulusTimestamps, instance=True),
            # stimulus_timestamps_to_json_ret
            {"sync_file": "sync.h5"},
            # raises
            "RunningAcquisition DataObject lacks information about",
            # expected
            None
        ),
        # Test to_json without stimulus_timestamps
        (
            # stimulus_file
            create_autospec(BehaviorStimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # stimulus_timestamps_to_json_ret
            None,
            # sync_file_to_json_ret
            None,
            # raises
            "RunningAcquisition DataObject lacks information about",
            # expected
            None
        ),
    ]
)
def test_running_acquisition_to_json(
    stimulus_file, stimulus_file_to_json_ret,
    stimulus_timestamps, stimulus_timestamps_to_json_ret, raises, expected
):
    if stimulus_file is not None:
        stimulus_file.to_json.return_value = stimulus_file_to_json_ret
    if stimulus_timestamps is not None:
        stimulus_timestamps._sync_file = create_autospec(SyncFile,
                                                         instance=True)
        stimulus_timestamps._monitor_delay = 0.0
        stimulus_timestamps._sync_file.to_json.return_value = (
            stimulus_timestamps_to_json_ret
        )

    running_acq = RunningAcquisition(
        running_acquisition=None,
        stimulus_file=stimulus_file,
        stimulus_timestamps=stimulus_timestamps
    )

    if raises:
        with pytest.raises(RuntimeError, match=raises):
            _ = running_acq.to_json()
    else:
        obt = running_acq.to_json()
        assert obt == expected


# Fixtures:
# nwbfile:
#   test/brain_observatory/behavior/conftest.py
# data_object_roundtrip_fixture:
#   test/brain_observatory/behavior/data_objects/conftest.py
@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("running_acq_data", [
    (
        # expected_running_acq_df
        pd.DataFrame(
            {
                "timestamps": [2.0, 4.0],
                "dx": [10.0, 12.0],
                "v_sig": [14.0, 16.0],
                "v_in": [18.0, 20.0]
            }
        ).set_index("timestamps")
    ),
])
def test_running_acquisition_nwb_roundtrip(
    nwbfile, data_object_roundtrip_fixture, roundtrip, running_acq_data
):
    running_acq = RunningAcquisition(running_acquisition=running_acq_data)
    nwbfile = running_acq.to_nwb(nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(nwbfile, RunningAcquisition)
    else:
        obt = RunningAcquisition.from_nwb(nwbfile)

    pd.testing.assert_frame_equal(
        obt.value, running_acq_data, check_like=True
    )
