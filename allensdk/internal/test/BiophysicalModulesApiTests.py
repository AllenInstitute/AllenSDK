from allensdk.api.api import Api
from allensdk.internal.api.queries.biophysical_module_api \
    import BiophysicalModuleApi

import unittest
from mock import MagicMock

class BiophysicalModulesApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BiophysicalModulesApiTests, self).__init__(*args, **kwargs)
        self.host = 'http://axon:3000'
    
    
    def setUp(self):
        Api.default_api_url = 'http://axon:3000'
        self.bma = BiophysicalModuleApi()
    
    
    def tearDown(self):
        self.bma = None
    
    
    def test_get_neuronal_model_runs(self):
        expected = "http://axon:3000/api/v2/data/query.json?q=model::NeuronalModelRun,rma::criteria,[id$in464137111],rma::include,well_known_files(well_known_file_type),neuronal_model(well_known_files(well_known_file_type),specimen(project,specimen_tags,ephys_roi_result(ephys_qc_criteria,well_known_files(well_known_file_type)),neuron_reconstructions(well_known_files(well_known_file_type)),ephys_sweeps(ephys_sweep_tags,ephys_stimulus(ephys_stimulus_type))),neuronal_model_template(neuronal_model_template_type,well_known_files(well_known_file_type))),rma::options[num_rows$eq'all'][count$eqfalse]"
        
        neuronal_model_run_id = 464137111
        
        self.bma.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.bma.get_neuronal_model_runs(neuronal_model_run_id)
        
        self.bma.json_msg_query.assert_called_once_with(expected)
        
    def test_get_neuronal_models(self):
        expected = "http://axon:3000/api/v2/data/query.json?q=model::NeuronalModel,rma::criteria,[id$in329322394],rma::include,well_known_files(well_known_file_type),specimen(project,specimen_tags,ephys_roi_result(ephys_qc_criteria,well_known_files(well_known_file_type)),neuron_reconstructions(well_known_files(well_known_file_type)),ephys_sweeps(ephys_sweep_tags,ephys_stimulus(ephys_stimulus_type))),neuronal_model_template(neuronal_model_template_type,well_known_files(well_known_file_type)),rma::options[num_rows$eq'all'][count$eqfalse]"
        
        neuronal_model_id = 329322394
        
        self.bma.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        self.bma.get_neuronal_models(neuronal_model_id)
        
        self.bma.json_msg_query.assert_called_once_with(expected)