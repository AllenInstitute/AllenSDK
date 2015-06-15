from allensdk.api.queries.structure.structure_graph_api import StructureGraphApi

import unittest, json
from mock import patch, mock_open

class StructureGraphApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StructureGraphApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.sa = StructureGraphApi()
    
    
    def tearDown(self):
        pass
    
    
    def test_structure_graph_1(self):
        expected = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
        
        structure_graph_id = 1
        actual = self.sa.build_structure_graph_query(structure_graph_id)
        
        self.assertEqual(expected, actual)
