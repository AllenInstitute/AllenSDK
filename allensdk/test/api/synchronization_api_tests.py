import unittest, json
from mock import patch, mock_open
from allensdk.api.queries.synchronization_api import SynchronizationApi

class SynchronizationApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SynchronizationApiTests, self).__init__(*args, **kwargs)
        self.ida = None
    
    
    def setUp(self):
        self.sa = SynchronizationApi()
    
    
    def tearDown(self):
        self.ida = None
    
    
    def test_api_doc_url_image_to_image(self):
        '''
        Notes
        -----
        Expected link is slightly modified for json and float serialization of zeros.
        
        See: `Image Alignment `<http://help.brain-map.org/display/mouseconnectivity/API#API-ImageAlignment>_
        , link labeled 'Sync a VISp and VISal experiment to a location in a SCs SectionDataSet'.
        '''
        expected = 'http://api.brain-map.org/api/v2/image_to_image/114754496.json?x=18232.000000&y=10704.000000&section_data_set_ids=113887162,116903968'
        
        section_image_id = 114754496
        (x, y) = (18232, 10704)
        section_data_set_ids = [113887162, 116903968]
        actual = self.sa.build_image_to_image_query(section_image_id,
                                                    x, y,
                                                    section_data_set_ids)
        
        self.assertEqual(actual, expected)
    
    
    def test_api_doc_url_image_to_atlas(self):
        '''
        Notes
        -----
        Expected link is slightly modified for json and float serialization of zeros.
        
        See: `Image Alignment `<http://help.brain-map.org/display/mouseconnectivity/API#API-ImageAlignment>_
        , link labeled 'Sync the P56 coronal reference atlas to a location in the SCs SectionDataSet'.
        '''
        expected = 'http://api.brain-map.org/api/v2/image_to_atlas/114754496.json?x=18232.000000&y=10704.000000&atlas_id=1'
        
        section_image_id = 114754496
        (x, y) = (18232, 10704)
        atlas_id = 1
        actual = self.sa.build_image_to_atlas_query(section_image_id,
                                                    x, y,
                                                    atlas_id)
        
        self.assertEqual(actual, expected)
