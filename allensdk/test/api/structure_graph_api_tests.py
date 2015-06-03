from allensdk.api.queries.structure_graph.structure_graph_api import StructureGraphApi

import unittest, json
from mock import patch, mock_open

class StructureGraphApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StructureGraphApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        pass
    
    
    def tearDown(self):
        pass
    
    
    def test_structure_graph_1(self):
        pass
    
if '__main__' == __name__:
    sga = StructureGraphApi()
    print(json.dumps(sga.get_structure_graph_by_id(1), indent=2))