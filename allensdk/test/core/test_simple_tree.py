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

from allensdk.core.simple_tree import SimpleTree

@pytest.fixture
def tree():

    nodes = [{'id': 0, 'parent': None}, {'id': 1, 'parent': 0}, 
             {'id': 2, 'parent': 0}, {'id': 3, 'parent': 1}, 
             {'id': 4, 'parent': 1}, {'id': 5, 'parent': 2}]
            
    parent_fn = lambda node: node['parent']
    id_fn = lambda node: node['id']
    
    return SimpleTree(nodes, id_fn, parent_fn)
    
    
def test_initialization(tree):

    assert( None in tree._parent_ids.values() )
    assert( len(tree._child_ids) == 6 )
    
    
def test_filter_nodes(tree):
    
    two_par = tree.filter_nodes(lambda node: node['parent'] == 2)
    assert( two_par[0]['id'] == 5 )
    assert( len(two_par) == 1 )
    
    
def test_value_map(tree):
    
    parent_map = tree.value_map(lambda node: node['id'], 
                                lambda node: node['parent'])
                                
    assert( len(parent_map) == 6 )
    assert( parent_map[2] == 0 )
    assert( parent_map[3] == 1 ) 
    
    
def test_node_ids(tree):

    obtained = tree.node_ids()
    expected = range(6)
     
    assert( set(obtained) == set(expected) )  
    
    
def test_parent_id(tree):

    nodes = [5, 4, 2]
    obtained = tree.parent_id(nodes)
    
    assert( allclose([2, 1, 0], obtained) )
    
    
def test_child_ids(tree):

    obtained = tree.child_ids([1])
    assert( set(obtained[0]) == set([4, 3]) )
    assert( len(obtained) == 1 )
    
    
def test_ancestor_ids(tree):

    obtained = tree.ancestor_ids([5, 1])
    
    assert( len(obtained) == 2 )
    assert( set(obtained[0]) == set([5, 2, 0]) )
    assert( set(obtained[1]) == set([1, 0]) )
    
    
def test_descendant_ids(tree):

    obtained = tree.descendant_ids([0, 3])

    assert( len(obtained) == 2 )
    assert( set(obtained[0]) == set(range(6)) )
    assert( set(obtained[1]) == set([3]) )
    
    
def test_node(tree):
    
    obtained = tree.node([0, 1])
    
    assert( len(obtained) == 2 )
    assert( obtained[0]['parent'] is None )
    assert( obtained[1]['id'] == 1 )
    
    
def test_node_default(tree):

    obtained = tree.node()
    assert( len(obtained) == 6 )
    

def test_parent(tree):

    obtained = tree.parent([0, 1])
    assert( len(obtained) == 2 )
    assert( obtained[0] is None )

def test_children(tree):
    
    obtained = tree.children([0, 5])

    assert( len(obtained) == 2 )
    assert( set(obtained[1]) == set([]) )
    assert( len(obtained[0]) == 2 ) 
    assert( isinstance(obtained[0][0], dict) )
    
def test_descendants(tree):

    obtained = tree.descendants([0, 3])

    assert( len(obtained) == 2 )
    assert( len(obtained[0]) == 6 )
    assert( obtained[1][0]['id'] == 3 )
    assert( isinstance(obtained[0][0], dict) )


def test_ancestors(tree):

    obtained = tree.ancestors([5, 1])
    
    assert( len(obtained) == 2 )
    assert( len(obtained[0]) == 3 )
    assert( isinstance(obtained[0][0], dict) )
    assert( len(obtained[1]) == 2 )
