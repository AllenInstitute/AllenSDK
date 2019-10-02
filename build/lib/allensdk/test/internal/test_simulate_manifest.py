from allensdk.internal.api.queries.biophysical_module_reader import \
    BiophysicalModuleReader
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
  "id": 8888,
  "storage_directory": "/neuronal/model/run/storage/directory",
  "neuronal_model": {
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
                "workflow_state": "auto_passed",
                "ephys_stimulus": {
                    "ephys_stimulus_type": {
                        "name": "Test"
                    }
                }
            },
            {
                "sweep_number": 2,
                "workflow_state": "manual_passed",
                "ephys_stimulus": {
                    "ephys_stimulus_type": {
                        "name": "Unknown"
                    }
                }
            },
            {
                "sweep_number": 3,
                "workflow_state": "failed",
                "ephys_stimulus": {
                    "ephys_stimulus_type": {
                        "name": "Long Square"
                    }
                }
            }
        ]
    },
    "neuronal_model_template": {
        "name": "Biophysical - perisomatic",
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
      },
      {
        "well_known_file_type": {
            "id": 475137571,
            "name": "NWB"
        },
        "well_known_file_type_id": 475137571,
        "id": 343434,
        "storage_directory": "/path/to/nwb_file/roi_maybe",
        "filename": "stimulus_input.nwb"
      }
    ]
  },
  "well_known_files": [
      {
        "well_known_file_type": {
            "id": 478840678,
            "name": "NWB_UNCOMPRESSED"
        },
        "well_known_file_type_id": 478840678,
        "id": 343434,
        "storage_directory": "/neuronal/model/run/dir",
        "filename": "pre_existing_output.nwb"
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
        
        manifest_string = output.getvalue()
        print(manifest_string)
        
        return manifest_string


@pytest.fixture
def no_param_config():
    json_data = json.loads(LIMS_MESSAGE_ONE_PARAM_FILE)
    json_data['well_known_files'] = []
    lims_message_no_param_file = json.dumps(json_data)
    scr = BiophysicalModuleReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=lims_message_no_param_file)):
        scr.read_lims_file(lims_json_path)
    
    return scr


@pytest.fixture
def one_param_config():
    scr = BiophysicalModuleReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_ONE_PARAM_FILE)):
        scr.read_lims_file(lims_json_path)
    
    return scr


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
    assert set(one_param_manifest_dict.keys()) == \
        set(['biophys', 'runs', 'neuron', 'manifest'])


def test_manifest_hoc(one_param_manifest_dict):
    assert set(one_param_manifest_dict['neuron'][0].keys()) == set(['hoc'])


def test_neuronal_model_run_id(one_param_manifest_dict):
    assert one_param_manifest_dict['runs'][0]['neuronal_model_run_id'] == 8888


def test_sweeps(one_param_manifest_dict):
    assert set(one_param_manifest_dict['runs'][0]['sweeps']) == set([1, 2])


def test_sweeps_by_type(one_param_manifest_dict):
    sweeps_by_type = one_param_manifest_dict['runs'][0]['sweeps_by_type']
    assert set(sweeps_by_type['Test']) == set([1])
    assert set(sweeps_by_type['Unknown']) == set([2])
    assert set(sweeps_by_type['Long Square']) == set([3])
    assert len(sweeps_by_type.keys()) == 3


def test_mod_file_paths(one_param_config):
    assert set(one_param_config.mod_file_paths()) == \
        set(['/path/to/mod_files/mod_file_1.mod'])


def test_update_well_known_file_not_existing(no_param_config):
    nwb_uncompressed_file_type = 478840678
    no_param_config.update_well_known_file('/path/to/pre_existing_output.nwb')
    wkf = no_param_config.lims_update_data['well_known_files'][0]
    assert 'id' not in wkf
    assert wkf['storage_directory'] == '/path/to'
    assert wkf['filename'] == 'pre_existing_output.nwb'
    assert wkf['well_known_file_type_id'] == nwb_uncompressed_file_type


def test_update_well_known_file_existing(one_param_config):
    nwb_uncompressed_file_type = 478840678
    one_param_config.update_well_known_file('/neuronal/model/run/dir/pre_existing_output.nwb')
    wkf = one_param_config.lims_update_data['well_known_files'][0]
    assert wkf['id'] == 343434
    assert wkf['storage_directory'] == '/neuronal/model/run/dir'
    assert wkf['filename'] == 'pre_existing_output.nwb'
    assert wkf['well_known_file_type_id'] == nwb_uncompressed_file_type


def test_update_well_known_file_existing_name_mismatch(one_param_config):
    nwb_uncompressed_file_type = 478840678
    one_param_config.update_well_known_file(
        '/neuronal/model/run/dir/8888_virtual_experiment.nwb')
    wkf = one_param_config.lims_update_data['well_known_files'][0]
    assert 'id' not in wkf
    assert wkf['storage_directory'] == '/neuronal/model/run/dir'
    assert wkf['filename'] == '8888_virtual_experiment.nwb'
    assert wkf['well_known_file_type_id'] == nwb_uncompressed_file_type


def test_manifest_keys(one_param_manifest_dict):
    expected_keys = set(['BASEDIR', 'WORKDIR', 'MORPHOLOGY', 'CODE_DIR',
                         'MODFILE_DIR', 'MOD_FILE_mod_file_1', 'stimulus_path',
                         'manifest', 'output_path', 'fit_parameters',
                         'neuronal_model_run_data', 'fit_parameters'])
     
    actual_keys = set([e['key'] for e in one_param_manifest_dict['manifest']])
    assert actual_keys == expected_keys
