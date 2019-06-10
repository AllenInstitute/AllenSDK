import subprocess as sp
import os
import json

import pytest

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