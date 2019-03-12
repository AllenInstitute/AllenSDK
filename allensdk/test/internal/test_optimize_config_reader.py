from allensdk.internal.api.queries.optimize_config_reader import \
    OptimizeConfigReader
import pytest
from mock import patch, mock_open
try:
    import __builtin__ as builtins
except:
    import builtins


LIMS_MESSAGE_NO_PARAM_FILES = """
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

LIMS_MESSAGE_ONE_PARAM_FILE = """
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
        "well_known_file_type": {
          "created_at": "2015-02-13T11:41:41-08:00",
          "id": 329230374,
          "name": "NeuronalModelParameters",
          "updated_at": "2015-02-13T11:41:41-08:00"
        },
        "well_known_file_type_id": 329230374,
        "id": 22222        
      }      
    ]
}
"""

LIMS_MESSAGE_TWO_PARAM_FILES = """
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
        "id": 22222        
      },
      {
        "well_known_file_type": {
          "created_at": "2015-02-13T11:41:41-08:00",
          "id": 329230374,
          "name": "NeuronalModelParameters",
          "updated_at": "2015-02-13T11:41:41-08:00"
        },
        "well_known_file_type_id": 329230374,
        "id": 22222        
      }      
    ]
}
"""


@pytest.fixture
def no_param_config():
    ocr = OptimizeConfigReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_NO_PARAM_FILES)):
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
def two_param_config():
    ocr = OptimizeConfigReader()

    lims_json_path = 'lims_message.json'
    
    with patch(builtins.__name__ + ".open",
               mock_open(read_data=LIMS_MESSAGE_TWO_PARAM_FILES)):
        ocr.read_lims_file(lims_json_path)
    
    return ocr


def test_no_params(no_param_config):
    assert no_param_config.lims_data['well_known_files'][0]['well_known_file_type']['id'] != 329230374
    no_param_config.update_well_known_file('/path/to/params_fit.json',
                                           OptimizeConfigReader.NEURONAL_MODEL_PARAMETERS)
    assert no_param_config.lims_update_data['well_known_files'][1]['well_known_file_type_id'] == 329230374
    

def test_one_param(one_param_config):
    assert one_param_config.lims_data['well_known_files'][1]['well_known_file_type']['id'] == 329230374
    one_param_config.update_well_known_file('/path/to/params_fit.json',
                                            OptimizeConfigReader.NEURONAL_MODEL_PARAMETERS)
    assert one_param_config.lims_update_data['well_known_files'][1]['well_known_file_type_id'] == 329230374
    assert one_param_config.lims_update_data['well_known_files'][1]['id'] == 22222


def test_two_params(two_param_config):
    assert two_param_config.lims_data['well_known_files'][1]['well_known_file_type']['id'] == 329230374
    two_param_config.update_well_known_file('/path/to/params_fit.json',
                                            OptimizeConfigReader.NEURONAL_MODEL_PARAMETERS)
    assert two_param_config.lims_update_data['well_known_files'][1]['well_known_file_type_id'] == 329230374
    assert two_param_config.lims_update_data['well_known_files'][1]['id'] == 22222
    assert len(two_param_config.lims_update_data['well_known_files']) == 2
