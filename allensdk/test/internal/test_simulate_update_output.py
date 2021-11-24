from allensdk.internal.api.queries.biophysical_module_reader import \
    BiophysicalModuleReader
import pytest
from mock import patch, mock_open
try:
    import __builtin__ as builtins
except:
    import builtins


LIMS_MESSAGE_NO_NWB_FILES = """
{
    "neuronal_model": {},
    "well_known_files": [
      {
          "well_known_file_type": {
            "created_at": "2013-11-05T16:59:04-08:00",
            "id": 292178729,
            "name": "BiophysicalModelDescription",
            "updated_at": "2013-11-05T16:59:04-08:00"
          },
        "well_known_file_type_id": 292178729,
        "id": 11111
      }
    ]
}
"""


LIMS_MESSAGE_ONE_NWB_FILE = """
{
    "neuronal_model": {},
    "well_known_files": [
      {
          "well_known_file_type": {
            "created_at": "2013-11-05T16:59:04-08:00",
            "id": 292178729,
            "name": "BiophysicalModelDescription",
            "updated_at": "2013-11-05T16:59:04-08:00"
          },
        "well_known_file_type_id": 292178729,
        "id": 11111
      },
      {
          "filename": "496537307_virtual_experiment.nwb",
          "id": 22222,
          "storage_directory": "/projects/mousecelltypes/vol1/prod572/neuronal_model_run_496537307/",
          "well_known_file_type": {
            "created_at": "2015-06-09T15:21:33-07:00",
            "id": 478840678,
            "name": "NWBUncompressed",
            "updated_at": "2015-06-09T15:21:33-07:00"
          },
          "well_known_file_type_id": 478840678
      }
    ]
}
"""


LIMS_MESSAGE_TWO_NWB_FILES = """
{
    "neuronal_model": {},
    "well_known_files": [
      {
          "well_known_file_type": {
            "created_at": "2013-11-05T16:59:04-08:00",
            "id": 292178729,
            "name": "BiophysicalModelDescription",
            "updated_at": "2013-11-05T16:59:04-08:00"
          },
        "well_known_file_type_id": 292178729,
        "id": 11111        
      },
      {
          "filename": "496537307_virtual_experiment.nwb",
          "id": 22222,
          "storage_directory": "/projects/mousecelltypes/vol1/prod572/neuronal_model_run_496537307/",
          "well_known_file_type": {
            "created_at": "2015-06-09T15:21:33-07:00",
            "id": 478840678,
            "name": "NWBUncompressed",
            "updated_at": "2015-06-09T15:21:33-07:00"
          },
          "well_known_file_type_id": 478840678
      },
      {
          "filename": "496537307_virtual_experiment.nwb",
          "id": 33333,
          "storage_directory": "/projects/mousecelltypes/vol1/prod572/neuronal_model_run_496537307/",
          "well_known_file_type": {
            "created_at": "2015-06-09T15:21:33-07:00",
            "id": 478840678,
            "name": "NWBUncompressed",
            "updated_at": "2015-06-09T15:21:33-07:00"
          },
          "well_known_file_type_id": 478840678
      }      
    ]
}
"""


@pytest.fixture
def no_nwb_config():
    bmr = BiophysicalModuleReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_NO_NWB_FILES)):
        bmr.read_lims_file(lims_json_path)
    
    return bmr


@pytest.fixture
def one_nwb_config():
    bmr = BiophysicalModuleReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_ONE_NWB_FILE)):
        bmr.read_lims_file(lims_json_path)
    
    return bmr


@pytest.fixture
def two_nwb_config():
    bmr = BiophysicalModuleReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_TWO_NWB_FILES)):
        bmr.read_lims_file(lims_json_path)
    
    return bmr


def test_no_nwb(no_nwb_config):
    assert no_nwb_config.lims_data['well_known_files'][0]['well_known_file_type']['id'] != 478840678
    no_nwb_config.update_well_known_file('/path/to/example.nwb')
    assert no_nwb_config.lims_update_data['well_known_files'][1]['well_known_file_type_id'] == 478840678
    


def test_one_nwb(one_nwb_config):
    assert one_nwb_config.lims_data['well_known_files'][1]['well_known_file_type']['id'] == 478840678
    one_nwb_config.update_well_known_file('/projects/mousecelltypes/vol1/prod572/neuronal_model_run_496537307/496537307_virtual_experiment.nwb')
    assert one_nwb_config.lims_update_data['well_known_files'][1]['well_known_file_type_id'] == 478840678
    assert one_nwb_config.lims_update_data['well_known_files'][1]['id'] == 22222


def test_two_nwb(two_nwb_config):
    assert two_nwb_config.lims_data['well_known_files'][1]['well_known_file_type']['id'] == 478840678
    two_nwb_config.update_well_known_file('/projects/mousecelltypes/vol1/prod572/neuronal_model_run_496537307/496537307_virtual_experiment.nwb')
    assert two_nwb_config.lims_update_data['well_known_files'][1]['well_known_file_type_id'] == 478840678
    assert two_nwb_config.lims_update_data['well_known_files'][1]['id'] == 22222
    assert len(two_nwb_config.lims_update_data['well_known_files']) == 2
