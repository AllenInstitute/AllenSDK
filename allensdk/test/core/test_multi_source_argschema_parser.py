import subprocess as sp
import os
import json

import pytest


@pytest.mark.parametrize('command,json_data,expected', [
    [
        'python {} --source lims --an_int 12 --a_float 0.2', 
        None,
        {'an_int': 12,  'a_float': 0.2, 'a_string': 'hello world', 'log_level': 'ERROR', 'not_in_lims': 9}
    ],
    [
        'python {} --source lims --an_int 12 --a_float 0.2 --a_string goodbye', 
        None,
        {'an_int': 12,  'a_float': 0.2, 'a_string': 'goodbye', 'log_level': 'ERROR', 'not_in_lims': 9}
    ],
    [
        'python {} --source lims --an_int 12 --a_float 0.2', 
        {'not_in_lims': 21},
        {'log_level': 'ERROR', 'not_in_lims': 21}
    ]
])
def test_source(command, json_data, expected, tmpdir_factory):

    if json_data is not None:
        temp_dir = str(tmpdir_factory.mktemp('multi_source_parser_test'))
        json_data_path = os.path.join(temp_dir, 'input_json.json')
        with open(json_data_path, 'w') as jsf:
            json.dump(json_data, jsf, indent=2)
        command = command + ' --input_json {}'.format(json_data_path)
        expected['input_json'] = json_data_path

    fixture_path = os.path.join(os.path.dirname(__file__), 'multi_source_argschema_parsing_fixture.py')
    command = command.format(fixture_path)
    returned = sp.check_output(command.split())

    obtained = json.loads(returned)
    for key, obt_val in obtained.items():
        assert obt_val == expected[key]
    assert set(obtained.keys()) == set(expected.keys())
