from __future__ import division, print_function, absolute_import
import re

import numpy as np

from simple_tree import SimpleTree



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
    def get_structures_by_id(self, *sids):
        return self.filter_nodes(lambda x: x['id'] in sids)
        
    
    def get_structures_by_name(self, *names):
        return self.filter_nodes(lambda x: x['safe_name'] in names)
        
        
    def get_structures_by_acronym(self, *acronyms):
        return self.filter_nodes(lambda x: x['acronym'] in acronyms)
        
        
    def get_colormap(self):
        return self.value_map(lambda x: x['id'], 
                              lambda y: y['color_hex_triplet'])
        
        
    def get_ancestor_id_map(self):
        return self.value_map(lambda x: x['id'], 
                              lambda y: self.get_ancestor_ids(y))
        
        
    def structure_descends_from(self, child_id, parent_id):
        return parent_id in self.ancestor_ids(child_id)
        
        
    @classmethod
    def from_ontologies_api(cls, oapi, insert_structure_sets=True):
                            
        structures = oapi.get_structures(1)
        if insert_structure_sets:
            
        
        
