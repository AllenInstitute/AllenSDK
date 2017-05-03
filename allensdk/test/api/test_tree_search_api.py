####test AllenSDK tree search api for Specimen and Structure
from allensdk.api.queries.tree_search_api import TreeSearchApi
import pytest
import json
from mock import MagicMock

@pytest.fixture
def tree_search():
    tsa = TreeSearchApi()
    tsa.json_msg_query = MagicMock(name='json_msg_query')

    return tsa

def test_get_specimen_tree(tree_search):
    ####ancestor true for Specimen
    kind = 'Specimen'
    db_id = 113817886
    ancestors = True
    descendants = None
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Specimen/113817886.json?ancestors=true")
    
    ####ancestor true for Specimen
    kind = 'Specimen'
    db_id = 113817886
    ancestors = True
    descendants = False
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Specimen/113817886.json?ancestors=true&descendants=false")
    
    ####ancestor false for Specimen
    kind = 'Specimen'
    db_id = 113817886
    ancestors = False
    descendants = True
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Specimen/113817886.json?ancestors=false&descendants=true")

def test_get_structure_tree(tree_search):
    ####ancestor True for Structure
    kind = 'Structure'
    db_id = 12547
    ancestors = True
    descendants = True
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Structure/12547.json?ancestors=true&descendants=true")
    
    ####ancestor False for Structure
    kind = 'Structure'
    db_id = 12547
    ancestors = False
    descendants = True
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Structure/12547.json?ancestors=false&descendants=true")
    
    ####ancestor None for Structure
    kind = 'Structure'
    db_id = 12547
    ancestors = None
    descendants = None
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Structure/12547.json")
