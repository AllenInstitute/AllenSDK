import os
from pathlib import Path

import pytest
import numpy as np


DATA_DIR = os.environ.get(
    "ECEPHYS_PIPELINE_DATA",
    os.path.join("/", "allen", "aibs", "informatics", "module_test_data", "ecephys", "extract_running_speed"),
)


def reparent(path, new_parent):
    return str(Path(new_parent) / Path(path).name)


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('input_json_fname,output_json_fname,exp_fname,exp_times_fname', [
    [
        "ECEPHYS_EXTRACT_RUNNING_SPEED_QUEUE_789848216_input.json", 
        'ECEPHYS_EXTRACT_RUNNING_SPEED_QUEUE_789848216_output.json', 
        '789848216_running_speeds.npy', 
        '789848216_running_timestamps.npy'
    ]
])
def test_extract_running_speed_module(
    use_temp_dir, input_json_fname, output_json_fname, exp_fname, exp_times_fname
):

    def renamer(input_json_data, data_dir, temp_dir):
        input_json_data['sync_h5_path'] = reparent(input_json_data['sync_h5_path'], data_dir)
        input_json_data['stimulus_pkl_path'] = reparent(input_json_data['stimulus_pkl_path'], data_dir)

        input_json_data['output_running_speeds_path'] = reparent(input_json_data['output_running_speeds_path'], temp_dir)
        input_json_data['output_timestamps_path'] = reparent(input_json_data['output_timestamps_path'], temp_dir)

        return input_json_data

    output_json_data = use_temp_dir(
        DATA_DIR, 'test_extract_running_speed', input_json_fname, output_json_fname, 
        'allensdk.brain_observatory.ecephys.extract_running_speed', renamer
    )

    expected = np.load(os.path.join(DATA_DIR, exp_fname), allow_pickle=False)
    expected_times = np.load(os.path.join(DATA_DIR, exp_times_fname), allow_pickle=False)

    obtained_running_speeds = np.load(output_json_data['output_running_speeds_path'], allow_pickle=False)
    obtained_timestamps = np.load(output_json_data['output_timestamps_path'], allow_pickle=False)

    assert np.allclose(expected, obtained_running_speeds)
    assert np.allclose(expected_times, obtained_timestamps)

