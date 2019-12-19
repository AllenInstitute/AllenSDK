""" Tests for the executable that synchronizes distinct data streams within an
ophys experiment. For tests of the logic used by this executable, see 
test_time_sync
"""

import os
import json
from typing import NamedTuple

import pytest
import numpy as np
import h5py

import allensdk
from allensdk.internal.pipeline_modules.run_ophys_time_sync import (
    TimeSyncOutputs, TimeSyncWriter, check_stimulus_delay, run_ophys_time_sync
)


@pytest.fixture
def outputs():
    return TimeSyncOutputs(
        100,
        0.35,
        0,
        1,
        2,
        3,
        np.linspace(0, 1, 10),
        np.linspace(1, 2, 10),
        np.linspace(2, 3, 10),
        np.linspace(3, 4, 10),
        np.arange(10),
        np.arange(10, 20),
        np.arange(20, 30)
    )

@pytest.fixture
def writer(tmpdir_factory):
    tmpdir_path = str(tmpdir_factory.mktemp("run_ophys_time_sync_tests"))
    return TimeSyncWriter(
        os.path.join(tmpdir_path, "data.h5"),
        os.path.join(tmpdir_path, "output.json")
    )


def test_validate_paths_writable(writer):
    try:
        writer.validate_paths()
    except Exception as err:
        pytest.fail(f"expected no error. Got: {err.__class__.__name__}(\"{err}\")")


@pytest.mark.parametrize("h5_key,expected", [
    ["stimulus_alignment", np.arange(10)],
    ["eye_tracking_alignment", np.arange(10, 20)],
    ["body_camera_alignment", np.arange(20, 30)],
    ["twop_vsync_fall", np.linspace(0, 1, 10)],
    ["ophys_delta", 0],
    ["stim_delta", 1],
    ["stim_delay", 0.35],
    ["eye_delta", 2],
    ["behavior_delta", 3]
])
def test_write_output_h5(writer, outputs, h5_key, expected):
    
    writer.write_output_h5(outputs)

    with h5py.File(writer.output_h5_path, "r") as obtained_file:
        obtained = obtained_file[h5_key]
        
        if isinstance(expected, np.ndarray):
            assert np.allclose(obtained, expected)
        else:
            assert obtained.value == expected


@pytest.mark.parametrize("json_key,expected", [
    ["allensdk_version", allensdk.__version__],
    ["experiment_id", 100],
    ["ophys_delta", 0],
    ["stim_delta", 1],
    ["stim_delay", 0.35],
    ["eye_delta", 2],
    ["behavior_delta", 3]
])
def test_write_output_json(writer, outputs, json_key, expected):

    writer.write_output_json(outputs)

    with open(writer.output_json_path, "r") as jf:
        obtained_dict = json.load(jf)
        obtained = obtained_dict[json_key]

        assert obtained == expected


@pytest.mark.parametrize("obt", np.linspace(0, 1, 4))
@pytest.mark.parametrize("mn", np.linspace(0, 1, 4))
@pytest.mark.parametrize("mx", np.linspace(0, 1, 4))
def test_check_stimulus_delay(obt, mn, mx):

    if obt < mn or obt > mx:
        with pytest.raises(ValueError):
            check_stimulus_delay(obt, mn, mx)
    else:
        check_stimulus_delay(obt, mn, mx)


def test_run_ophys_time_sync():

    class Aligner(NamedTuple):
        corrected_stim_timestamps: np.ndarray
        corrected_ophys_timestamps: np.ndarray
        corrected_eye_video_timestamps: np.ndarray
        corrected_behavior_video_timestamps: np.ndarray
    
    aligner = Aligner(
        (np.arange(10), 0, 0.5), 
        (np.arange(10), 1), 
        (np.arange(10), 2), 
        (np.arange(10), 3)
    )

    obtained = run_ophys_time_sync(aligner, 100, 0.0, 2.0)

    # store mismatches in an array so we can show every distinct failure
    mismatches = []
    for name, expected in [
        ["experiment_id", 100],
        ["stimulus_delay", 0.5],
        ["ophys_delta", 1],
        ["stimulus_delta", 0],
        ["eye_delta", 2],
        ["behavior_delta", 3],
        ["ophys_times", np.arange(10)],
        ["stimulus_times", np.arange(10)],
        ["eye_times", np.arange(10)],
        ["behavior_times", np.arange(10)],
        ["stimulus_alignment", np.arange(10)],
        ["eye_alignment", np.arange(10)],
        ["behavior_alignment", np.arange(10)]
    ]:

        current_obt = getattr(obtained, name)

        if isinstance(expected, np.ndarray):
            match = np.allclose(expected, current_obt)
        else:
            match = expected == current_obt

        if not match:
            mismatches.append(
                f"{name} mismatched: expected {expected}, "
                f"obtained {current_obt}"
            )

    assert len(mismatches) == 0, "\n" + "\n".join(mismatches)