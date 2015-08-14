import unittest, json
from mock import patch, mock_open
from allensdk.api.queries.grid_data_api import GridDataApi

class GridDataApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GridDataApiTests, self).__init__(*args, **kwargs)
        self.ida = None
    
    
    def setUp(self):
        self.gda = GridDataApi()
    
    
    def tearDown(self):
        self.ida = None
    
    
    def test_api_doc_url_download_expression_grid(self):
        '''Url to download the 200um density volume for the Mouse Brain Atlas SectionDataSet 69816930.
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        , example 'Download the 200um density volume for the Mouse Brain Atlas SectionDataSet 69816930'.
        '''
        expected = 'http://api.brain-map.org/grid_data/download/69816930'
        
        section_data_set_id = 69816930
        actual = self.gda.build_expression_grid_download_query(section_data_set_id)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_download_expression_grid_energy_intensity(self):
        '''Url to download the 200um energy and intensity volumes for Mouse Brain Atlas SectionDataSet 69816930.
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        , example 'Download the 200um energy and intensity volumes for Mouse Brain Atlas SectionDataSet 69816930'.
        
        The id in the example url doesn't match the caption.
        '''
        expected = 'http://api.brain-map.org/grid_data/download/183282970?include=energy,intensity'
        
        section_data_set_id = 183282970
        include = ['energy', 'intensity']
        actual = self.gda.build_expression_grid_download_query(section_data_set_id,
                                                               include=include)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_projection_grid(self):
        '''Url to download the 100um density volume for the Mouse Connectivity Atlas SectionDataSet 181777177.
        
        Notes
        -----
        See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA>`_
        , example 'Download the 100um density volume for the Mouse Connectivity Atlas SectionDataSet 181777177'.
        '''
        expected = 'http://api.brain-map.org/grid_data/download_file/181777177'
        
        section_data_set_id = 181777177
        actual = self.gda.build_projection_grid_download_query(section_data_set_id)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_projection_grid_injection_fraction_resolution(self):
        '''Url to download the 25um injection_fraction volume for Mouse Connectivity Atlas SectionDataSet 181777177.
        
        Notes
        -----
        See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA>`_
        , example 'Download the 25um injection_fraction volume for Mouse Connectivity Atlas SectionDataSet 181777177'.
        '''
        expected = 'http://api.brain-map.org/grid_data/download_file/181777177?image=injection_fraction&resolution=25'
        
        section_data_set_id = 181777177
        actual = self.gda.build_projection_grid_download_query(section_data_set_id,
                                                               image=['injection_fraction'],
                                                               resolution=25)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_experiment_id(self):
        '''Url to search for relevant experiments' IDs (SectionDataSets)
        for the Mouse Brain Atlas' coronal Adora2a experiment.
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        and `Example Queries for Experiment Metadata <http://help.brain-map.org/display/api/Example+Queries+for+Experiment+Metadata#ExampleQueriesforExperimentMetadata-MouseBrain>`_
        for additional documentation.
        '''
        expected = "http://api.brain-map.org/api/v2/data/query.json?q=model::SectionDataSet,rma::criteria,[failed$eqfalse],products[abbreviation$eq'Mouse'],plane_of_section[name$eq'coronal'],genes[acronym$eq'Adora2a']"
        
        product_abbreviation = 'Mouse'
        plane_of_section = 'coronal'
        gene_acronym = 'Adora2a'
        actual = self.gda.build_experiment_id_url(product_abbreviation,
                                                  plane_of_section,
                                                  gene_acronym)
        
        self.assertEqual(actual, expected)


