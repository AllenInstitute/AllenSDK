import os
from pathlib import Path
import json
import subprocess as sp

import pytest
import pandas as pd

@pytest.fixture
def use_temp_dir(tmpdir_factory):
    def fn(data_dir, tempdir_name, input_json_fname, output_json_fname, module, renamer_cb):

        temp_dir = str(tmpdir_factory.mktemp(tempdir_name))
            
        input_json_path = os.path.join(data_dir, input_json_fname)
        new_input_json_path = os.path.join(temp_dir, input_json_fname)

        output_json_path = os.path.join(temp_dir, output_json_fname)

        with open(input_json_path, 'r') as input_json:
            input_json_data = json.load(input_json)

        input_json_data = renamer_cb(input_json_data, data_dir, temp_dir)

        with open(new_input_json_path, 'w') as new_input_json:
            json.dump(input_json_data, new_input_json)

        sp.check_call([
            'python', '-m', module, 
            '--input_json', new_input_json_path, 
            '--output_json', output_json_path
        ])

        with open(output_json_path, 'r') as output_json:
            output_json_data = json.load(output_json)

        return output_json_data

    return fn


DATA_DIR = os.environ.get(
    "ECEPHYS_PIPELINE_DATA",
    os.path.join("/", "allen", "aibs", "informatics", "module_test_data", "ecephys", "extract_running_speed"),
)


def reparent(path, new_parent):
    return str(Path(new_parent) / Path(path).name)


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('input_json_fname,output_json_fname,exp_fname', [
    [
        "ECEPHYS_EXTRACT_RUNNING_SPEED_QUEUE_744228101_input.json", 
        'ECEPHYS_EXTRACT_RUNNING_SPEED_QUEUE_744228101ls_output.json', 
        '744228101_running_speeds.h5', 
    ]
])
def test_extract_running_speed_module(
    use_temp_dir, input_json_fname, output_json_fname, exp_fname
):

    def renamer(input_json_data, data_dir, temp_dir):
        input_json_data['sync_h5_path'] = reparent(input_json_data['sync_h5_path'], data_dir)
        input_json_data['stimulus_pkl_path'] = reparent(input_json_data['stimulus_pkl_path'], data_dir)

        input_json_data['output_path'] = reparent(input_json_data['output_path'], temp_dir)

        return input_json_data

    output_json_data = use_temp_dir(
        DATA_DIR, 'test_extract_running_speed', input_json_fname, output_json_fname, 
        'allensdk.brain_observatory.extract_running_speed', renamer
    )

    expected_path = os.path.join(DATA_DIR, exp_fname)
    expected_velos = pd.read_hdf(expected_path, key="running_speed")
    expected_raw = pd.read_hdf(expected_path, key="raw_data")

    obtained_velos = pd.read_hdf(output_json_data['output_path'], key="running_speed")
    obtained_raw = pd.read_hdf(output_json_data['output_path'], key="raw_data")

    pd.testing.assert_frame_equal(expected_velos, obtained_velos, check_like=True)
    pd.testing.assert_frame_equal(expected_raw, obtained_raw, check_like=True)

