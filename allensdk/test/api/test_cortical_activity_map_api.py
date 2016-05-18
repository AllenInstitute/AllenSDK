from allensdk.internal.api.queries.cortical_activity_map_api \
    import CorticalActivityMapApi

import pytest
from mock import MagicMock
    
    
@pytest.fixture
def cam_api():
    cam = CorticalActivityMapApi('http://testwarehouse:9000')
    cam.json_msg_query = MagicMock(name='json_msg_query')
    
    return cam

    
def test_list_isi_experiments(cam_api):
    cam_api.list_isi_experiments()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::IsiExperiment,rma::options[num_rows$eq'all'][count$eqfalse]"    
    cam_api.json_msg_query.assert_called_once_with(expected)
        
def test_get_isi_experiments(cam_api):      
    isi_experiment_id = 503316697     
    cam_api.get_isi_experiments(isi_experiment_id)

    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::IsiExperiment,rma::criteria,[id$in503316697],rma::include,experiment_container(ophys_experiments,targeted_structure),rma::options[num_rows$eq'all'][count$eqfalse]"
    cam_api.json_msg_query.assert_called_once_with(expected)
 
 
def test_get_ophys_experiments_one_id(cam_api):
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::OphysExperiment,rma::criteria,[id$in502066273],rma::include,well_known_files(well_known_file_type),rma::options[num_rows$eq'all'][count$eqfalse]"
      
    ophys_experiment_id = 502066273
    cam_api.get_ophys_experiments(ophys_experiment_id)     
    cam_api.json_msg_query.assert_called_once_with(expected)
     
     
def test_get_experiment_container_metric(cam_api):
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamExperimentContainerMetric,rma::criteria,[id$in511510627],rma::options[num_rows$eq'all'][count$eqfalse]"
      
    id = 511510627
      
    cam_api.get_experiment_container_metric(id)
    cam_api.json_msg_query.assert_called_once_with(expected)


def test_get_experiment_container(cam_api):
    id = 511510753
    cam_api.get_experiment_container(id)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ExperimentContainer,rma::criteria,[id$in511510753],rma::include,ophys_experiments,isi_experiment,specimen,targeted_structure,rma::options[count$eqfalse]"
    cam_api.json_msg_query.assert_called_once_with(expected)
 
     
def test_get_column_definitions(cam_api):      
    api_class_name = cam_api.quote_string('ApiTbiDonorMetric')
    cam_api.get_column_definitions(api_class_name=api_class_name)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiColumnDefinition,rma::criteria,[api_class_name$eq'ApiTbiDonorMetric'],rma::options[num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)


def test_list_column_definition_class_names(cam_api):              
    cam_api.list_column_definition_class_names()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiColumnDefinition,rma::options[only$eq'api_class_name'][num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)
 
 
def test_get_stimulus_mapping_no_ids(cam_api):
    cam_api.get_stimulus_mapping()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::criteria,,rma::options[num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)
 
 
def test_get_stimulus_mapping_one_id(cam_api):
    ids = 15
    cam_api.get_stimulus_mapping(stimulus_mapping_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::criteria,[stimulus_mapping_id$in15],rma::options[num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)    
 
 
def test_get_stimulus_mapping_one_id(cam_api):
    ids = 15
    cam_api.get_stimulus_mapping(stimulus_mapping_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::criteria,[id$in15],rma::options[num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)    


def test_get_stimulus_mapping_two_ids(cam_api):
    ids = [15, 43]
    cam_api.get_stimulus_mapping(stimulus_mapping_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::criteria,[id$in15,43],rma::options[num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)    
 

def test_get_cell_metric_no_ids(cam_api):
    cam_api.get_cell_metric()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamCellMetric,rma::criteria,,rma::options[num_rows$eq'all'][count$eqfalse]"     
    cam_api.json_msg_query.assert_called_once_with(expected)    


def test_get_cell_metric_one_ids(cam_api):
    id = 517394843
    cam_api.get_cell_metric(cell_specimen_ids=id)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamCellMetric,rma::criteria,[cell_specimen_id$in517394843],rma::options[num_rows$eq'all'][count$eqfalse]"      
    cam_api.json_msg_query.assert_called_once_with(expected)    


def test_get_cell_metric_two_ids(cam_api):
    ids = [517394843,517394850]
    cam_api.get_cell_metric(cell_specimen_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamCellMetric,rma::criteria,[cell_specimen_id$in517394843,517394850],rma::options[num_rows$eq'all'][count$eqfalse]"     
    cam_api.json_msg_query.assert_called_once_with(expected)    
