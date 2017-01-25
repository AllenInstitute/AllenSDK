from __future__ import division, print_function, absolute_import
import re
import operator as op

import numpy as np

from .simple_tree import SimpleTree


class StructureTree( SimpleTree ):

    # These structure sets may be of interest
    # check descriptions at:
    # http://api.brain-map.org/api/v2/data/StructureSet/query.json?
    STRUCTURE_SETS = {114512892: 'Mouse Connectivity - BDA/AAV Primary Injection Structures',  
                      112905813: 'Mouse Connectivity - BDA/AAV All Injection Structures', 
                      167587189: 'Mouse Connectivity - Summary',  
                      112905828: 'Mouse Connectivity - Projection All Injection Structures',  
                      184527634: 'Mouse Connectivity - Target Search', 
                      3: 'Mouse - Areas', 
                      396673091: 'Mouse Cell Types - Structures', 
                      514166994: 'CAM targeted structure set', 
                      2: 'Mouse - Coarse', 
                      114512891: 'Mouse Connectivity - Projection Primary Injection Structures'}
   
    # These fields will be removed from each structure when building from 
    # the api. 
    # unsure about: hemisphere_id, failed, failed_facet, structure_name_facet, safe_name
    # st_level might be needed for devmouse
    DELETE_FIELDS = ['atlas_id', 'ontology_id', 'sphinx_id', 'weight', 
                     'depth', 'parent_structure_id', 
                     'neuro_name_structure_id', 
                     'neuro_name_structure_id_path', 'st_level']

    def __init__(self, nodes):
        '''A tree whose nodes are brain structures and whose edges indicate 
        physical containment.
        
        Parameters
        ----------
        nodes : list of dict
            Each specifies a structure. Fields are:
            
            'acronym' : str
                Abbreviated name for the structure.
            'atlas_id' : 
            
        
        '''
        
        super(StructureTree, self).__init__(nodes,
                                            lambda s: int(s['id']),
                                            lambda s: s['structure_id_path'][-2] \
                                                if s['structure_id_path'] is not None \
                                                and len(s['structure_id_path']) > 1 \
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
        
        overlap = lambda x: (set(structure_set_ids) & set(x['structure_sets']))
        return self.filter_nodes(overlap)
        
        
    def get_colormap(self):
        '''Get a dictionary mapping structure ids to colors across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are strings containing RGB-order 
            hexidecimal color.
        
        '''
    
        return self.value_map(lambda x: x['id'], 
                              lambda y: y['color_hex_triplet'])
        
        
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
        
        
    def has_overlaps(self, structure_ids):
        '''Determine if a list of structures contains spatial overlaps
        
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
    def from_ontologies_api(oapi, graph_id=1, structure_set_ids=None):
        '''Construct a StructureTree from an OntologiesApi instance.
        
        Parameters
        ----------
        oapi : OntologiesApi
            Used to download structures and find structure sets.
        graph_id : int, optional
            Specifies the structure graph from which to draw structures. 
            Default is the mouse brain atlas.
        structure_set_ids : list of int, optional
            Each structure in the tree will be given an additional key 
            ("structure_sets") that maps to a list of structure sets 
            containing that structure. Available sets are specified by this 
            parameter. If none are supplied, a general default list will be 
            used.
        
        Returns
        -------
        StructureTree
        
        '''
                            
        structures = oapi.get_structures(graph_id)
        
        if structure_set_ids is None:
            structure_set_ids = StructureTree.STRUCTURE_SETS.keys()
            
        sts_map = oapi.get_structure_set_map(structure_sets=structure_set_ids)
        
        for ii, val in enumerate(structures):
        
            val = delete_fields(val, *StructureTree.DELETE_FIELDS)
                                
            val['structure_sets'] = sts_map[val['id']]
            
            val['structure_id_path'] = [int(stid) for stid 
                                        in val['structure_id_path'].split('/')
                                        if stid != ''] 
        
            structures[ii] = val
        
        return StructureTree(structures)
        
        
def delete_fields(dictionary, *field_names):
    return {k:v for k, v in dictionary.iteritems() if not k in field_names}
    
