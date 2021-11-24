# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import pytest
import mock
from numpy import allclose

from allensdk.core.simple_tree import SimpleTree


@pytest.fixture
def tree():

    s = frozenset([1, 2, 3])

    nodes = [{'id': 0, 'parent': None, 1: 2, s: 'a'}, {'id': 1, 'parent': 0, 1: 7, s: 'd'}, 
             {'id': 2, 'parent': 0, 1: 3, s: 'b'}, {'id': 3, 'parent': 1, 1: 6, s: 'e'}, 
             {'id': 4, 'parent': 1, 1: 4, s: 'c'}, {'id': 5, 'parent': 2, 1: 5, s: 'f'}]
            
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


@pytest.mark.parametrize('key,val,to,exp', [ ['id', [2, 1, 3], lambda x: x['id'],[2, 1, 3]],
                                             [lambda x: x['id'], [2, 1, 3], lambda x: x['id'],[2, 1, 3]],
                                             [1, [3, 7, 6], lambda x: x['id'],[2, 1, 3]],
                                             [frozenset([1, 2, 3]), ['b'], lambda x: x[1], [3]] ])
def test_nodes_by_property(tree, key, val, to, exp):

    obt = tree.nodes_by_property( key, val, to_fn=to )
    assert( allclose( obt, exp) )

    
def test_value_map(tree):
    
    parent_map = tree.value_map(lambda node: node['id'], 
                                lambda node: node['parent'])
                                
    assert( len(parent_map) == 6 )
    assert( parent_map[2] == 0 )
    assert( parent_map[3] == 1 ) 
    

def test_value_map_nonunique(tree):
    
    with pytest.raises( RuntimeError ):
        parent_map = tree.value_map(lambda node: node['parent'], 
                                    lambda node: node['id'])

    
def test_node_ids(tree):

    obtained = tree.node_ids()
    expected = range(6)
     
    assert( set(obtained) == set(expected) )  
    
    
def test_parent_ids(tree):

    nodes = [5, 4, 2]
    obtained = tree.parent_ids(nodes)
    
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
    
    
def test_nodes(tree):
    
    obtained = tree.nodes([0, 1])
    
    assert( len(obtained) == 2 )
    assert( obtained[0]['parent'] is None )
    assert( obtained[1]['id'] == 1 )
    
    
def test_nodes_default(tree):

    obtained = tree.nodes()
    assert( len(obtained) == 6 )
    

def test_parents(tree):

    obtained = tree.parents([0, 1])
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


def test_cbs(tree):

    nodes = tree.nodes()
    for node in nodes:
        assert( node['id'] == tree.node_id_cb(node) )
        assert( node['parent'] == tree.parent_id_cb(node) )
