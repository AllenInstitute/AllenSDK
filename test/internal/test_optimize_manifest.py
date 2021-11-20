from allensdk.internal.api.queries.optimize_config_reader import \
    OptimizeConfigReader
import pytest
from mock import patch, mock_open, MagicMock
from six import StringIO
from io import IOBase
import json
try:
    import __builtin__ as builtins
except:
    import builtins


LIMS_MESSAGE_ONE_PARAM_FILE = """
{
    "storage_directory": "storage directory 11111",
    "specimen_id": 98765,
    "specimen": {
        "neuron_reconstructions": [
          {
             "superseded": false,
             "manual": true,
             "well_known_files": [
               {
                   "well_known_file_type_id": 303941301,
                   "storage_directory": "/path/to/morphology",
                   "filename": "morphology_file.swc"
               }
             ]
          }
        ],
        "ephys_roi_result": {
             "well_known_files": [
               {
                   "well_known_file_type_id": 475137571,
                   "storage_directory": "/path/to/stimulus_nwb",
                   "filename": "stimulus_1234.nwb"
               }
            ]
        },
        "ephys_sweeps": [
            {
                "sweep_number": 1,
                "workflow_state": "auto_passed"
            },
            {
                "sweep_number": 2,
                "workflow_state": "manual_passed"
            },
            {
                "sweep_number": 3,
                "workflow_state": "failed"
            }
        ]
    },
    "neuronal_model_template": {
        "well_known_files": [
          {
              "well_known_file_type": {
                "created_at": "2013-11-05T16:59:04-08:00",
                "id": 292178729,
                "name": "BiophysicalModelDescription",
                "updated_at": "2013-11-05T16:59:04-08:00"
              },
            "well_known_file_type_id": 292178729,
            "id": 11111,
            "storage_directory": "/path/to/mod_files",
            "filename": "mod_file_1.mod"
          }
        ]
    },
    "well_known_files": [
      {
        "well_known_file_type": {
          "created_at": "2015-02-13T11:41:41-08:00",
          "id": 329230374,
          "name": "NeuronalModelParameters",
          "updated_at": "2015-02-13T11:41:41-08:00"
        },
        "well_known_file_type_id": 329230374,
        "id": 22222,
        "storage_directory": "/path/to/neuronal_model",
        "filename": "existing_well_known.file"
      }
    ]
}
"""


def manifest_as_string(reader):
    output = StringIO()

    with patch(builtins.__name__ + ".open",
               mock_open(),
               create=True) as manifest_f:
        manifest_f.return_value = MagicMock(spec=IOBase)
        file_handle = manifest_f.return_value.__enter__.return_value
        file_handle.write.side_effect = output.write
        
        reader.to_manifest("test_manifest.json")
        
        return output.getvalue()


@pytest.fixture
def no_param_config():
    json_data = json.loads(LIMS_MESSAGE_ONE_PARAM_FILE)
    json_data['well_known_files'] = []
    lims_message_no_param_file = json.dumps(json_data)
    ocr = OptimizeConfigReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=lims_message_no_param_file)):
        ocr.read_lims_file(lims_json_path)
    
    return ocr


@pytest.fixture
def one_param_config():
    ocr = OptimizeConfigReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_ONE_PARAM_FILE)):
        ocr.read_lims_file(lims_json_path)
    
    return ocr


@pytest.fixture
def one_param_manifest_dict(one_param_config):
    reader = one_param_config
    json_string = manifest_as_string(reader)
    the_dict = json.loads(json_string)
    
    return the_dict


def test_to_manifest(one_param_config):
    with patch(builtins.__name__ + ".open",
               mock_open(),
               create=True) as manifest_f:
        manifest_f.return_value = MagicMock(spec=IOBase)
        file_handle = manifest_f.return_value.__enter__.return_value
        one_param_config.to_manifest("test_manifest.json")
        manifest_f.assert_called_once_with("test_manifest.json", "wb+")


def test_top_level_keys(one_param_manifest_dict):
    assert set(one_param_manifest_dict.keys()) == set(['biophys', 'runs', 'neuron', 'manifest'])


def test_manifest_hoc(one_param_manifest_dict):
    assert set(one_param_manifest_dict['neuron'][0].keys()) == set(['hoc'])


def test_specimen_id(one_param_manifest_dict):
    assert one_param_manifest_dict['runs'][0]['specimen_id'] == 98765


def test_sweeps(one_param_manifest_dict):
    assert set(one_param_manifest_dict['runs'][0]['sweeps']) == set([1, 2])


def test_mod_file_paths(one_param_config):
    assert set(one_param_config.mod_file_paths()) == set(['/path/to/mod_files/mod_file_1.mod'])


def test_update_well_known_file_not_existing(no_param_config):
    fit_file_type = 329230374
    no_param_config.update_well_known_file('/path/to/new_fit.json')
    wkf = no_param_config.lims_update_data['well_known_files'][0]
    assert 'id' not in wkf
    assert wkf['storage_directory'] == '/path/to'
    assert wkf['filename'] == 'new_fit.json'
    assert wkf['well_known_file_type_id'] == fit_file_type


def test_update_well_known_file_existing(one_param_config):
    fit_file_type = 329230374
    one_param_config.update_well_known_file('/path/to/new_fit.json')
    wkf = one_param_config.lims_update_data['well_known_files'][0]
    assert wkf['id'] == 22222
    assert wkf['storage_directory'] == '/path/to'
    assert wkf['filename'] == 'new_fit.json'
    assert wkf['well_known_file_type_id'] == fit_file_type
    
    
def test_manifest_keys(one_param_manifest_dict):
    expected_keys = set(['BASEDIR', 'WORKDIR', 'MORPHOLOGY', 'MODFILE_DIR',
                         'MOD_FILE_mod_file_1', 'stimulus_path', 'manifest',
                         'output', 'neuronal_model_data', 'upfile', 'downfile',
                         'passive_fit_data', 'stage_1_jobs', 'fit_1_file', 
                         'fit_2_file', 'fit_3_file', 'fit_type_path',
                         'target_path', 'fit_config_json', 'final_hof_fit',
                         'final_hof', 'output_fit_file'])
    
    actual_keys = set([e['key'] for e in one_param_manifest_dict['manifest']])
    assert actual_keys == expected_keys
