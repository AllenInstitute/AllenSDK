import unittest, json
from mock import patch, mock_open
from allensdk.api.queries.image_download_api import ImageDownloadApi

class ImageDownloadApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ImageDownloadApiTests, self).__init__(*args, **kwargs)
        self.ida = None
    
    
    def setUp(self):
        self.ida = ImageDownloadApi()
    
    
    def tearDown(self):
        self.ida = None
    
    
    def test_api_doc_url_download_image_downsampled(self):
        '''
        Notes
        -----
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'Download image downsampled by factor of 6 using default thresholds'.
        '''
        expected = 'http://api.brain-map.org/api/v2/section_image_download/126862575?downsample=6&range=0,932,0,1279,0,4095'
        
        section_image_id = 126862575
        actual = self.ida.build_section_image_url(section_image_id,
                                                  downsample=6,
                                                  range=[0,932,  0,1279, 0,4095])
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_download_image_full_resolution(self):
        '''
        Notes
        -----
        See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
        , link labeled 'Download a region of interest at full resolution using default thresholds'.
        '''
        expected = 'http://api.brain-map.org/api/v2/section_image_download/126862575?left=19045&top=11684&width=1000&height=1000&range=0,932,0,1279,0,4095'
        
        section_image_id = 126862575
        actual = self.ida.build_section_image_url(section_image_id,
                                                  left=19045,
                                                  top=11684,
                                                  width=1000,
                                                  height=1000,
                                                  range=[0,932, 0,1279, 0,4095])
        
        self.assertEqual(actual, expected)
