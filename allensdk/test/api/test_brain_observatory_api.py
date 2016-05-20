from allensdk.api.queries.brain_observatory_api \
    import BrainObservatoryApi

import pytest
from mock import MagicMock
    
    
@pytest.fixture
def bo_api():
    bo = BrainObservatoryApi('http://testwarehouse:9000')
    bo.json_msg_query = MagicMock(name='json_msg_query')
    
    return bo

    
def test_list_isi_experiments(bo_api):
    bo_api.list_isi_experiments()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::IsiExperiment,rma::options[num_rows$eq'all'][count$eqfalse]"    
    bo_api.json_msg_query.assert_called_once_with(expected)
        
def test_get_isi_experiments(bo_api):      
    isi_experiment_id = 503316697     
    bo_api.get_isi_experiments(isi_experiment_id)

    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::IsiExperiment,rma::criteria,[id$in503316697],rma::include,experiment_container(ophys_experiments,targeted_structure),rma::options[num_rows$eq'all'][count$eqfalse]"
    bo_api.json_msg_query.assert_called_once_with(expected)
 
 
def test_get_ophys_experiments_one_id(bo_api):
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::OphysExperiment,rma::criteria,[id$in502066273],rma::include,well_known_files(well_known_file_type),targeted_structure,specimen(donor(age,transgenic_lines)),rma::options[num_rows$eq'all'][count$eqfalse]"
      
    ophys_experiment_id = 502066273
    bo_api.get_ophys_experiments(ophys_experiment_id)     
    bo_api.json_msg_query.assert_called_once_with(expected)
     
def test_get_experiment_container_metrics(bo_api):
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamExperimentContainerMetric,rma::criteria,[id$in511510627],rma::options[num_rows$eq'all'][count$eqfalse]"
      
    tid = 511510627
      
    bo_api.get_experiment_container_metrics(tid)
    bo_api.json_msg_query.assert_called_once_with(expected)


def test_get_experiment_containers(bo_api):
    tid = 511510753
    bo_api.get_experiment_containers(tid)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ExperimentContainer,rma::criteria,[id$in511510753],rma::include,ophys_experiments,isi_experiment,specimen(donor(age,transgenic_lines)),targeted_structure,rma::options[num_rows$eq'all'][count$eqfalse]"
    bo_api.json_msg_query.assert_called_once_with(expected)
 
     
def test_get_column_definitions(bo_api):      
    api_class_name = bo_api.quote_string('ApiTbiDonorMetric')
    bo_api.get_column_definitions(api_class_name=api_class_name)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiColumnDefinition,rma::criteria,[api_class_name$eq'ApiTbiDonorMetric'],rma::options[num_rows$eq'all'][count$eqfalse]"      
    bo_api.json_msg_query.assert_called_once_with(expected)


def test_list_column_definition_class_names(bo_api):              
    bo_api.list_column_definition_class_names()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiColumnDefinition,rma::options[only$eq'api_class_name'][num_rows$eq'all'][count$eqfalse]"      
    bo_api.json_msg_query.assert_called_once_with(expected)
 
 
def test_get_stimulus_mappings_no_ids(bo_api):
    bo_api.get_stimulus_mappings()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::options[num_rows$eq'all'][count$eqfalse]"      
    bo_api.json_msg_query.assert_called_once_with(expected)
 
 
def test_get_stimulus_mappings_one_id(bo_api):
    ids = 15
    bo_api.get_stimulus_mappings(stimulus_mapping_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::criteria,[id$in15],rma::options[num_rows$eq'all'][count$eqfalse]"      
    bo_api.json_msg_query.assert_called_once_with(expected)    


def test_get_stimulus_mappings_two_ids(bo_api):
    ids = [15, 43]
    bo_api.get_stimulus_mappings(stimulus_mapping_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamStimulusMapping,rma::criteria,[id$in15,43],rma::options[num_rows$eq'all'][count$eqfalse]"      
    bo_api.json_msg_query.assert_called_once_with(expected)    
 

def test_get_cell_metrics_no_ids(bo_api):
    bo_api.get_cell_metrics()
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamCellMetric,rma::options[num_rows$eq'all'][count$eqfalse]"     
    bo_api.json_msg_query.assert_called_once_with(expected)    


def test_get_cell_metrics_one_ids(bo_api):
    tid = 517394843
    bo_api.get_cell_metrics(cell_specimen_ids=tid)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamCellMetric,rma::criteria,[cell_specimen_id$in517394843],rma::options[num_rows$eq'all'][count$eqfalse]"      
    bo_api.json_msg_query.assert_called_once_with(expected)    


def test_get_cell_metrics_two_ids(bo_api):
    ids = [517394843,517394850]
    bo_api.get_cell_metrics(cell_specimen_ids=ids)
    expected = "http://testwarehouse:9000/api/v2/data/query.json?q=model::ApiCamCellMetric,rma::criteria,[cell_specimen_id$in517394843,517394850],rma::options[num_rows$eq'all'][count$eqfalse]"     
    bo_api.json_msg_query.assert_called_once_with(expected)    
