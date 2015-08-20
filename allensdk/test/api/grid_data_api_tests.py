import unittest
from mock import MagicMock
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
        path = '69816930.zip'
        
        section_data_set_id = 69816930
        
        self.gda.retrieve_file_over_http = \
            MagicMock(name='retrieve_file_over_http')
        
        self.gda.download_expression_grid_data(section_data_set_id)
        
        self.gda.retrieve_file_over_http.assert_called_once_with(expected, path)
    
    
    def test_api_doc_url_download_expression_grid_energy_intensity(self):
        '''Url to download the 200um energy and intensity volumes for Mouse Brain Atlas SectionDataSet 69816930.
        
        Notes
        -----
        See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
        , example 'Download the 200um energy and intensity volumes for Mouse Brain Atlas SectionDataSet 69816930'.
        
        The id in the example url doesn't match the caption.
        '''
        expected = 'http://api.brain-map.org/grid_data/download/183282970?include=energy,intensity'
        path = '183282970.zip'

        section_data_set_id = 183282970
        include = ['energy', 'intensity']
        
        self.gda.retrieve_file_over_http = \
            MagicMock(name='retrieve_file_over_http')
        
        self.gda.download_expression_grid_data(section_data_set_id,
                                               include=include)
        
        self.gda.retrieve_file_over_http.assert_called_once_with(expected, path)
    
    
    def test_api_doc_url_projection_grid(self):
        '''Url to download the 100um density volume for the Mouse Connectivity Atlas SectionDataSet 181777177.
        
        Notes
        -----
        See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA>`_
        , example 'Download the 100um density volume for the Mouse Connectivity Atlas SectionDataSet 181777177'.
        '''
        expected = 'http://api.brain-map.org/grid_data/download_file/181777177'
        path = '181777177.nrrd'

        self.gda.retrieve_file_over_http = \
            MagicMock(name='retrieve_file_over_http')
        
        section_data_set_id = 181777177
        self.gda.download_projection_grid_data(section_data_set_id)
        
        self.gda.retrieve_file_over_http.assert_called_once_with(expected, path)
    
    
    def test_api_doc_url_projection_grid_injection_fraction_resolution(self):
        '''Url to download the 25um injection_fraction volume for Mouse Connectivity Atlas SectionDataSet 181777177.
        
        Notes
        -----
        See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA>`_
        , example 'Download the 25um injection_fraction volume for Mouse Connectivity Atlas SectionDataSet 181777177'.
        '''
        expected = 'http://api.brain-map.org/grid_data/download_file/181777177?image=injection_fraction&resolution=25'

        self.gda.retrieve_file_over_http = \
            MagicMock(name='retrieve_file_over_http')
        
        section_data_set_id = 181777177
        path = 'id.nrrd'
        self.gda.download_projection_grid_data(section_data_set_id,
                                               [self.gda.INJECTION_FRACTION],
                                               resolution=25,
                                               save_file_path=path)
        
        self.gda.retrieve_file_over_http.assert_called_once_with(expected, path)