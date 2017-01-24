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

    def __init__(self, nodes):
        super(StructureTree, self).__init__(nodes,
                                            lambda s: int(s['id']),
                                            lambda s: int(s['parent_structure_id']) \
                                                if s['parent_structure_id'] is not None \
                                                and np.isfinite(s['parent_structure_id']) \
                                                else None)
                                                
                           
    # or just use _nodes keys                     
    def get_structures_by_id(self, sids):
        return self.filter_nodes(lambda x: x['id'] in sids)
        
    
    def get_structures_by_name(self, names):
        return self.filter_nodes(lambda x: x['safe_name'] in names)
        
        
    def get_structures_by_set_id(self, structure_set_ids):
        overlap = lambda x: set(ss_ids) & set(x['structure_set_ids'])
        return self.filter_nodes(overlap)
        
        
    def get_structures_by_acronym(self, acronyms):
        return self.filter_nodes(lambda x: x['acronym'] in acronyms)
        
        
    def get_colormap(self):
        return self.value_map(lambda x: x['id'], 
                              lambda y: y['color_hex_triplet'])
        
        
    def get_ancestor_id_map(self):
        return self.value_map(lambda x: x['id'], 
                              lambda y: self.ancestor_ids([y['id']])[0])
        
        
    def structure_descends_from(self, child_id, parent_id):
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
    def from_ontologies_api(oapi, structure_set_ids=None):
                            
        structures = oapi.get_structures(1)
        
        if structure_set_ids is None:
            structure_set_ids = StructureTree.STRUCTURE_SETS.keys()
            
        ss_map = oapi.get_structure_set_map(structure_sets=structure_set_ids)
        structures = map(lambda x: dict(x, structure_sets=ss_map[x['id']]), 
                         structures)
        
        return StructureTree(structures)
        
