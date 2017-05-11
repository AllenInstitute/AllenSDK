# Copyright 2017 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import pytest
import mock
from numpy import allclose

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree


@pytest.fixture
def nodes():

    return [{'id': 0, 'structure_id_path': [0], 'color_hex_triplet': '000000', 'acronym': 'rt', 'name': 'root', 'structure_set_ids':[1, 4]}, 
            {'id': 1, 'structure_id_path': [0, 1], 'color_hex_triplet': '000fff', 'acronym': 'a', 'name': 'alpha', 'structure_set_ids': [1, 3]}, 
            {'id': 2, 'structure_id_path': [0, 2], 'color_hex_triplet': 'ffffff', 'acronym': 'b', 'name': 'beta', 'structure_set_ids': [1, 2]}]


@pytest.fixture
def tree(nodes):
    return StructureTree(nodes)
    

@pytest.fixture
def oapi():
    oa = OntologiesApi()
    
    oa.get_structures = mock.MagicMock(return_value=[{'id': 1, 'structure_id_path': '1'}])
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
    

def test_get_structures_by_set_id(tree):

    obtained = tree.get_structures_by_set_id([2, 3])

    assert( len(obtained) == 2 )
    
    
def test_get_colormap(tree):
    
    obtained = tree.get_colormap()
    assert( allclose(obtained[0], [0, 0, 0]) )
    assert( allclose(obtained[2], [255, 255, 255]) )   
    
    
def test_get_name_map(tree):
    
    obtained = tree.get_name_map()
    assert( obtained[0] == 'root' )
    assert( obtained[2] == 'beta' )  
    
    
def test_get_id_acronym_map(tree):
    
    obtained = tree.get_id_acronym_map()
    assert( obtained['rt'] == 0 )
    

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


def test_clean_structures(nodes):

    dirty_node = {'id': 0, 'structure_id_path': '/0/', 
                  'color_hex_triplet': '000000', 'acronym': 'rt', 
                  'name': 'root', 'structure_sets':[{'id': 1}, {'id': 4}]}
                  
    clean_node = StructureTree.clean_structures([dirty_node])[0]
    assert( isinstance(clean_node['color_hex_triplet'], list) )
    assert( isinstance(clean_node['structure_id_path'], list) )
    
    
def test_clean_structures_no_sets():
    
    dirty_node = {'id': 0, 'structure_id_path': '/0/', 
                  'color_hex_triplet': '000000', 'acronym': 'rt', 
                  'name': 'root'}
    
    clean_node = StructureTree.clean_structures([dirty_node])
    st = StructureTree(clean_node)
    
    assert( len(clean_node[0]['structure_set_ids']) == 0 )
    
    
def test_clean_structures_only_ids():
    
    dirty_node = {'id': 0, 'structure_id_path': '/0/', 
                  'color_hex_triplet': '000000', 'acronym': 'rt', 
                  'name': 'root', 'structure_set_ids': [1, 2, 3] }
    
    clean_node = StructureTree.clean_structures([dirty_node])
    st = StructureTree(clean_node)
    
    assert( len(clean_node[0]['structure_set_ids']) == 3 )
    
    
def test_clean_structures_ids_sets():
    
    dirty_node = {'id': 0, 'structure_id_path': '/0/', 
                  'color_hex_triplet': '000000', 'acronym': 'rt', 
                  'name': 'root', 'structure_set_ids': [1, 2, 3], 
                  'structure_sets': [{'id': 1}, {'id': 4}] }
    
    clean_node = StructureTree.clean_structures([dirty_node])
    st = StructureTree(clean_node)
    
    assert( len(clean_node[0]['structure_set_ids']) == 4 )
    
    
def test_clean_structures_str_id():

    dirty_node = {'id': '0', 'structure_id_path': '/0/', 
                  'color_hex_triplet': '000000', 'acronym': 'rt', 
                  'name': 'root', 'structure_set_ids': [1, 2, 3], 
                  'structure_sets': [{'id': 1}, {'id': 4}] }
    
    clean_node = StructureTree.clean_structures([dirty_node])
    st = StructureTree(clean_node)
    
    assert( set(st.node_ids()) == set([0]) )
    
    
def test_get_structure_sets(tree):

    expected = set([1, 2, 3, 4])
    obtained = tree.get_structure_sets()
    assert( expected == obtained )


def test_clean_structures_weird_keys():
    
    dirty_node = {'id': 5, 'dummy_key': 'dummy_val'}
    clean_node = StructureTree.clean_structures([dirty_node])[0]

    assert( len(clean_node) == 2 )
    assert( clean_node['id'] == 5 )


@pytest.mark.parametrize('inp,out', [('990099', [153, 0, 153]), 
                                     ('#990099', [153, 0, 153]), 
                                     ([153, 0, 153], [153, 0, 153]), 
                                     ((153., 0., 153.), [153, 0, 153]), 
                                     ([long(153), long(0), long(153)], [153, 0, 153])])
def test_hex_to_rgb(inp, out):
    obt = StructureTree.hex_to_rgb(inp)
    assert(allclose(obt, out))


@pytest.mark.parametrize('inp,out', [('/1/2/3/', [1, 2, 3]),
                                     ('1/2/3/', [1, 2, 3]), 
                                     ('/1/2/3', [1, 2, 3]), 
                                     ('1/2/3', [1, 2, 3]), 
                                     ([1, 2, 3], [1, 2, 3]),
                                     ([1.0, long(2), 3], [1, 2, 3]), 
                                     ((1, 2, 3), [1, 2, 3]), 
                                     ('', [])])
def test_path_to_list(inp, out):
    obt = StructureTree.path_to_list(inp)
    assert(allclose(obt, out))
