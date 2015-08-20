import unittest
from mock import MagicMock
from allensdk.api.queries.synchronization_api import SynchronizationApi

class SynchronizationApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SynchronizationApiTests, self).__init__(*args, **kwargs)
        self.ida = None
    
    
    def setUp(self):
        self.sa = SynchronizationApi()
    
    
    def tearDown(self):
        self.ida = None
    
    
    def test_image_to_image(self):
        '''
        Notes
        -----
        Expected link is slightly modified for json and float serialization of zeros.
        
        See: `Image Alignment `<http://help.brain-map.org/display/mouseconnectivity/API#API-ImageAlignment>_
        , link labeled 'Sync a VISp and VISal experiment to a location in a SCs SectionDataSet'.
        '''
        expected = 'http://api.brain-map.org/api/v2/image_to_image/114754496.json?x=18232.000000&y=10704.000000&section_data_set_ids=113887162,116903968'

        self.sa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        section_image_id = 114754496
        (x, y) = (18232, 10704)
        section_data_set_ids = [113887162, 116903968]
        
        _ = self.sa.get_image_to_image(section_image_id,
                                       x, y,
                                       section_data_set_ids)
        
        self.sa.json_msg_query.assert_called_once_with(expected)


    def test_image_to_image_2d(self):
        expected = 'http://api.brain-map.org/api/v2/image_to_image_2d/68173101.json?x=6208.000000&y=2368.000000&section_image_ids=68173103,68173105,68173107'
                    
        self.sa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        section_image_id = 68173101
        (x, y) = (6208, 2368)
        section_image_ids = [68173103, 68173105, 68173107]
        
        _ = self.sa.get_image_to_image_2d(section_image_id,
                                          x, y,
                                          section_image_ids)
        
        self.sa.json_msg_query.assert_called_once_with(expected)
    

    def test_reference_to_image(self):
        expected = 'http://api.brain-map.org/api/v2/reference_to_image/10.json?x=6085.000000&y=3670.000000&z=4883.000000&section_data_set_ids=68545324,67810540'
                    
        self.sa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        reference_space_id = 10
        (x, y, z) = (6085, 3670, 4883)
        section_data_set_ids = [68545324, 67810540]
        
        _ = self.sa.get_reference_to_image(reference_space_id,
                                          x, y, z,
                                          section_data_set_ids)
        
        self.sa.json_msg_query.assert_called_once_with(expected)


    def test_image_to_reference(self):
        expected = 'http://api.brain-map.org/api/v2/image_to_reference/68173101.json?x=6208.000000&y=2368.000000'
                    
        self.sa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        section_image_id = 68173101
        (x, y) = (6208, 2368)
        
        _ = self.sa.get_image_to_reference(section_image_id,
                                          x, y)
        
        self.sa.json_msg_query.assert_called_once_with(expected)


    def test_structure_to_image(self):
        expected = 'http://api.brain-map.org/api/v2/structure_to_image/68545324.json?structure_ids=315,698,1089,703,477,803,512,549,1097,313,771,354'
                    
        self.sa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        section_data_set_id = 68545324
        structure_ids = [315,698,1089,703,477,803,512,549,1097,313,771,354]
        
        _ = self.sa.get_structure_to_image(section_data_set_id,
                                           structure_ids)
        
        self.sa.json_msg_query.assert_called_once_with(expected)

    
    def test_image_to_atlas(self):
        '''
        Notes
        -----
        Expected link is slightly modified for json and float serialization of zeros.
        
        See: `Image Alignment `<http://help.brain-map.org/display/mouseconnectivity/API#API-ImageAlignment>_
        , link labeled 'Sync the P56 coronal reference atlas to a location in the SCs SectionDataSet'.
        '''
        expected = 'http://api.brain-map.org/api/v2/image_to_atlas/114754496.json?x=18232.000000&y=10704.000000&atlas_id=1'

        self.sa.json_msg_query = \
            MagicMock(name='json_msg_query')
        
        section_image_id = 114754496
        (x, y) = (18232, 10704)
        atlas_id = 1
        _ = self.sa.get_image_to_atlas(section_image_id,
                                       x, y,
                                       atlas_id)
        
        self.sa.json_msg_query.assert_called_once_with(expected)
