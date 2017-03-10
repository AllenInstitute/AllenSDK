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


from __future__ import division, print_function, absolute_import
import re
import operator as op

import numpy as np

from .simple_tree import SimpleTree


class StructureTree( SimpleTree ):

    FIELDS = ['acronym', 'color_hex_triplet', 'graph_id', 'graph_order', 
              'id', 'name', 'structure_id_path', 'structure_set_ids']

    def __init__(self, nodes):
        '''A tree whose nodes are brain structures and whose edges indicate 
        physical containment.
        
        Parameters
        ----------
        nodes : list of dict
            Each specifies a structure. Fields are:
            
            'acronym' : str
                Abbreviated name for the structure.
            'color_hex_triplet' : str
                Canonical RGB hexidecimal color assigned to this structure
            'graph_id' : int
                Specifies the structure graph containing this structure.
            'graph_order' : int
                Canonical position in the flattened structure graph.
            'id': int
                Unique structure specifier.
            'name' : str
                Full name of structure.
            'structure_id_path' : list of int
                This structure's ancestors (inclusive) from the root of the 
                tree.
            'structure_set_ids' : list of int
                Unique identifiers of structure sets to which this structure 
                belongs.
        
        '''
        
        super(StructureTree, self).__init__(nodes,
                                            lambda s: int(s['id']),
                                            lambda s: s['structure_id_path'][-2] \
                                                if len(s['structure_id_path']) > 1 \
                                                and s['structure_id_path'] is not None \
                                                and np.isfinite(s['structure_id_path'][-2]) \
                                                else None)
                                                
                                             
    def get_structures_by_id(self, structure_ids):
        '''Obtain a list of brain structures from their structure ids
        
        Parameters
        ----------
        structure_ids : list of int
            Get structures corresponding to these ids.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
        
        '''
    
        return self.filter_nodes(lambda x: x['id'] in structure_ids)
        
    
    def get_structures_by_name(self, names):
        '''Obtain a list of brain structures from their names,
        
        Parameters
        ----------
        names : list of str
            Get structures corresponding to these names.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
            
        '''
        
        return self.filter_nodes(lambda x: x['name'] in names)
        
        
    def get_structures_by_acronym(self, acronyms):
        '''Obtain a list of brain structures from their acronyms
        
        Parameters
        ----------
        names : list of str
            Get structures corresponding to these acronyms.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
            
        '''
        
        return self.filter_nodes(lambda x: x['acronym'] in acronyms)
        
        
    def get_structures_by_set_id(self, structure_set_ids):
        '''Obtain a list of brain structures from by the sets that contain 
        them.
        
        Parameters
        ----------
        structure_set_ids : list of int
            Get structures belonging to these structure sets.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
            
        '''
        
        overlap = lambda x: (set(structure_set_ids) & set(x['structure_set_ids']))
        return self.filter_nodes(overlap)
        
        
    def get_colormap(self):
        '''Get a dictionary mapping structure ids to colors across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are RGB lists of integers.
        
        '''
    
        return self.value_map(lambda x: x['id'], 
                              lambda y: StructureTree.hex_to_rgb(y['color_hex_triplet']))

                              
                              
    def get_name_map(self):
        '''Get a dictionary mapping structure ids to names across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are structure name strings.
        
        '''
    
        return self.value_map(lambda x: x['id'], 
                              lambda y: y['name'])
        
        
    def get_id_acronym_map(self):
        '''Get a dictionary mapping structure acronyms to ids across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure acronyms. Values are structure ids.
        
        '''
        
        return self.value_map(lambda x: x['acronym'], 
                              lambda y: y['id'])
        
        
    def get_ancestor_id_map(self):
        '''Get a dictionary mapping structure ids to ancestor ids across all 
        nodes. 
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are lists of ancestor ids.
        
        '''

        return self.value_map(lambda x: x['id'], 
                              lambda y: self.ancestor_ids([y['id']])[0])
        
        
    def structure_descends_from(self, child_id, parent_id):
        '''Tests whether one structure descends from another. 
        
        Parameters
        ----------
        child_id : int
            Id of the putative child structure.
        parent_id : int
            Id of the putative parent structure.
            
        Returns
        -------
        bool :
            True if the structure specified by child_id is a descendant of 
            the one specified by parent_id. Otherwise False.
        
        '''
    
        return parent_id in self.ancestor_ids([child_id])[0]
    
    
    def get_structure_sets(self):
        '''Lists all unique structure sets that are assigned to at least one 
        structure in the tree.
        
        Returns
        -------
        list of int : 
            Elements are ids of structure sets.
        
        '''
        
        return set(reduce(op.add, map(lambda x: x['structure_set_ids'], 
                                      self.node())))
        
        
    def has_overlaps(self, structure_ids):
        '''Determine if a list of structures contains structures along with 
        their ancestors
        
        Parameters
        ----------
        structure_ids : list of int
            Check this set of structures for overlaps
            
        Returns
        -------
        set : 
            Ids of structures that are the ancestors of other structures in 
            the supplied set.
        
        '''
    
        ancestor_ids = reduce(op.add, 
                              map(lambda x: x[1:], 
                                  self.ancestor_ids(structure_ids)))
        return (set(ancestor_ids) & set(structure_ids))
        

    @staticmethod
    def clean_structures(structures, field_whitelist=None):
        '''Convert structures_with_sets query results into a form that can be 
        used to construct a StructureTree
        
        Parameters
        ----------
        structures : list of dict
            Each element describes a structure. Should have a structure id path 
            field (str values) and a structure_sets field (list of dict).
        field_whitelist : list of str, optional
           Each record should keep only fields in this list
           
        Returns
        -------
        list of dict : 
            structures, after conversion of structure_id_path and structure_sets 
        
        '''

        if field_whitelist is None:
            field_whitelist = StructureTree.FIELDS

        for ii, val in enumerate(structures):

            val['structure_id_path'] = [int(stid) for stid 
                                        in val['structure_id_path'].split('/')
                                        if stid != ''] 
   
            val['structure_set_ids'] = [sts['id'] for sts in val['structure_sets']]

            structures[ii] = StructureTree.filter_dict(val, *field_whitelist)

        return structures
        
        
    @staticmethod
    def hex_to_rgb(hex_color):
        '''Convert a hexadecimal color string to a uint8 triplet
        
        Parameters
        ----------
        hex_color : string 
            Must be 6 characters long, unless it is 7 long and the first 
            character is #.
            
        Returns
        -------
        list of int : 
            3 characters long - 1 per two characters in the input string.
        
        '''
        
        if hex_color[0] == '#':
            hex_color = hex_color[1:]
        
        return [int(hex_color[a * 2: a*2 + 2], 16) for a in xrange(3)]


    @staticmethod
    def filter_dict(dictionary, *pass_keys):
        return {k:v for k, v in dictionary.iteritems() if k in pass_keys}
    
