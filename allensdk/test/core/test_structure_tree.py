
import pytest
import mock
from numpy import allclose

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree


@pytest.fixture
def tree():

    nodes = [{'id': 0, 'parent_structure_id': None, 'color_hex_triplet': '000000', 'acronym': 'rt', 'safe_name': 'root'}, 
            {'id': 1, 'parent_structure_id': 0, 'color_hex_triplet': '000fff', 'acronym': 'a', 'safe_name': 'alpha'}, 
            {'id': 2, 'parent_structure_id': 0, 'color_hex_triplet': 'ffffff', 'acronym': 'b', 'safe_name': 'beta'}, ]
            
    return StructureTree(nodes)
    

@pytest.fixture
def oapi():
    oa = OntologiesApi()
    
    oa.get_structures = mock.MagicMock(return_value=[{'id': 1, 'parent_structure_id': None}])
    oa.get_structure_set_map = mock.MagicMock(return_value={1: [2, 3]})
    
    return oa
    
    
def test_get_structures_by_id(tree):
    
    obtained = tree.get_structures_by_id([1, 2])
    assert( len(obtained) == 2 ) 
    
    
def test_get_structures_by_name(tree):
    
    obtained = tree.get_structures_by_name(['root'])
    assert( len(obtained) == 1 )
    
    
def test_get_structures_by_acronym(tree):

    obtained = tree.get_structures_by_acronym(['rt', 'a', 'b'])
    assert( len(obtained) == 3)
    
    
def test_get_colormap(tree):
    
    obtained = tree.get_colormap()
    assert( obtained[0] == '000000' )
    assert( obtained[2] == 'ffffff' )   
    

def test_get_ancestor_id_map(tree):

    obtained = tree.get_ancestor_id_map()
    assert( set(obtained[2]) == set([2, 0]) )
    

def test_structure_descends_from(tree):
    
    assert( tree.structure_descends_from(2, 0) )
    assert( not tree.structure_descends_from(0, 1) )
    
    
def test_has_overlaps(tree):
    
    obtained = tree.has_overlaps([0, 1, 2])
    assert( obtained == set([0]) )
    
    obag = tree.has_overlaps([1, 2])
    assert( not obag )
    
    
def test_from_ontologies_api(oapi):
    
    st = StructureTree.from_ontologies_api(oapi)
    
    oapi.get_structures.assert_called_with(1)
    oapi.get_structure_set_map.assert_called_with(structure_sets=StructureTree.STRUCTURE_SETS.keys())
    
    assert(len(st._nodes) == 1)
