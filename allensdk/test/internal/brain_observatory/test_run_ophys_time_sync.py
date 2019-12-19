""" Tests for the executable that synchronizes distinct data streams within an
ophys experiment. For tests of the logic used by this executable, see 
test_time_sync
"""

import os
import json

import pytest
import numpy as np
import h5py

import allensdk
from allensdk.internal.pipeline_modules.run_ophys_time_sync import (
    TimeSyncOutputs, TimeSyncWriter, run_ophys_time_sync
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


