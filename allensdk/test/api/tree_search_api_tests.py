from allensdk.api.queries.tree_search.tree_search_api import TreeSearchApi

import unittest, json
from mock import patch, mock_open

class TreeSearchApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TreeSearchApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        pass
    
    
    def tearDown(self):
        pass
    
    
    def test_structure_graph_1(self):
        pass
    
if '__main__' == __name__:
    tsa = TreeSearchApi()
    #print(json.dumps(tsa.get_structure_tree_by_id(695), indent=2))
    #print(json.dumps(tsa.get_structure_tree_by_id(695, descendants=True), indent=2))
    print(json.dumps(tsa.get_specimen_tree_by_id(113817886, ancestors=True), indent=2))
