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


import functools
import operator as op
from collections import defaultdict
from six import iteritems

class SimpleTree( object ):
    def __init__(self, nodes, 
                 node_id_cb, 
                 parent_id_cb):
        '''A tree structure
        
        Parameters
        ----------
        nodes : list of dict
            Each dict is a node in the tree. The keys of the dict name the 
            properties of the node and should be consistent across nodes.
        node_id_cb : function | node dict -> node id
            Calling node_id_cb on a node dictionary ought to produce a unique 
            identifier for that node (we call this the node's id). The type 
            of the node id is up to you, but ought to be consistent across 
            nodes and must be hashable.
        parent_id_cb : function | node_dict => parent node's id
            As node_id_cb, but returns the id of the node's parent.
            
        Notes
        -----
        It is easy to pass a pandas DataFrame as the nodes. Just use the 
        to_dict method of the dataframe like so:
            list_of_dict = your_dataframe.to_dict('record')
            your_tree = SimpleTree(list_of_dict, ...)
        Converting a list of dictionaries to a pandas DataFrame is also very 
        easy. The DataFrame constructor does it for you:
            your_dataframe = pandas.DataFrame(list_of_dict)
             
        '''

        self._nodes = defaultdict(lambda *x: None, { node_id_cb(n):n for n in nodes })
        self._parent_ids = defaultdict(lambda *x: None, { nid:parent_id_cb(n) for nid,n in iteritems(self._nodes) })
        self._child_ids = defaultdict(lambda *x: None, { nid:[] for nid in self._nodes })

        for nid in self._parent_ids:
            pid = self._parent_ids[nid]
            if pid is not None:
                self._child_ids[pid].append(nid)


    def filter_nodes(self, criterion):
        '''Obtain a list of nodes filtered by some criterion
        
        Parameters
        ----------
        criterion : function | node dict => bool
            Only nodes for which criterion returns true will be returned.
            
        Returns
        -------
        list of dict :
            Items are node dictionaries that passed the filter.
        
        '''
    
        return list(filter(criterion, self._nodes.values()))
        
        
    def value_map(self, from_fn, to_fn):
        '''Obtain a look-up table relating a pair of node properties across 
        nodes
        
        Parameters
        ----------
        from_fn : function | node dict => hashable value
            The keys of the output dictionary will be obtained by calling 
            from_fn on each node.
        to_fn : function | node_dict => value
            The values of the output function will be obtained by calling 
            to_fn on each node.
            
        Returns
        -------
        dict :
            Maps the node property defined by from_fn to the node property 
            defined by to_fn across nodes.
            
        Notes
        -----
        The resulting map is not necessarily 1-to-1! 
        
        '''
    
        return {from_fn(v): to_fn(v) for v in self._nodes.values()}


    def node_ids(self):
        '''Obtain the node ids of each node in the tree
        
        Returns
        -------
        list :
            elements are node ids 
        
        '''
    
        return self._nodes.keys()


    def parent_id(self, node_ids):
        '''Obtain the ids of one or more nodes' parents
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose parents you wish to find.
        
        Returns
        -------
        list of hashable : 
            Items are ids of input nodes' parents in order.
        
        '''
        
        return [ self._parent_ids[nid] for nid in node_ids ]

   
    def child_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' children
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose children you wish to find.
            
        Returns
        -------
        list of list of hashable : 
            Items are lists of input nodes' children's ids.
            
        '''
    
        return [ self._child_ids[nid] for nid in node_ids ]


    def ancestor_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' ancestors
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose ancestors you wish to find.
        
        Returns
        -------
        list of list of hashable : 
            Items are lists of input nodes' ancestors' ids.
        
        Notes
        -----
        Given the tree:
        A -> B -> C
         `-> D
          
        The ancestors of C are [C, B, A]. The ancestors of A are [A]. The 
        ancestors of D are [D, A]
        
        '''
    
        out = []
        for nid in node_ids:
        
            current = [nid]
            while current[-1] is not None:
                current.extend(self.parent_id([current[-1]]))
            out.append(current[:-1])
                
        return out
            
    
    def descendant_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' descendants
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose descendants you wish to find.
        
        Returns
        -------
        list of list of hashable : 
            Items are lists of input nodes' descendants' ids.
        
        Notes
        -----
        Given the tree:
        A -> B -> C
         `-> D
          
        The descendants of A are [B, C, D]. The descendants of C are [].
        
        '''
    
        out = []
        for ii, nid in enumerate(node_ids):
        
            current = [nid]
            children = self.child_ids([nid])[0]
            
            if children:
                current.extend(functools.reduce(op.add, map(list, 
                               self.descendant_ids(children))))
                               
            out.append(current)
        return out
            

    def node(self, node_ids=None):
        '''Get one or more nodes' full dictionaries from their ids.
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes to be returned. Default is all.
            
        Returns
        -------
        list of dict : 
            Items are nodes corresponding to argued ids.
        '''
    
        if node_ids is None:
            node_ids = self.node_ids()
    
        return [ self._nodes[nid] for nid in node_ids ]


    def parent(self, node_ids):
        '''Get one or mode nodes' parent nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose parents will be found.
            
        Returns
        -------
        list of dict : 
            Items are parents of nodes corresponding to argued ids.
        
        '''
        
        return self.node([self._parent_ids[nid] for nid in node_ids])


    def children(self, node_ids):
        '''Get one or mode nodes' child nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose children will be found.
            
        Returns
        -------
        list of list of dict : 
            Items are lists of child nodes corresponding to argued ids.
        
        '''
    
        return list(map(self.node, self.child_ids(node_ids)))


    def descendants(self, node_ids):
        '''Get one or mode nodes' descendant nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose descendants will be found.
            
        Returns
        -------
        list of list of dict : 
            Items are lists of descendant nodes corresponding to argued ids.
        
        '''
        
        return list(map(self.node, self.descendant_ids(node_ids)))

    
    def ancestors(self, node_ids):
        '''Get one or mode nodes' ancestor nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose ancestors will be found.
            
        Returns
        -------
        list of list of dict : 
            Items are lists of ancestor nodes corresponding to argued ids.
        
        '''
    
        return list(map(self.node, self.ancestor_ids(node_ids)))
    
