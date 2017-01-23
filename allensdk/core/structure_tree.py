from __future__ import division, print_function, absolute_import
import re

import numpy as np

from simple_tree import SimpleTree



class StructureTree( SimpleTree ):
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
        
        
    def get_cortical_layers(self, layer=None):
    
        if layer is not None:
            pattern_string = 'layer(.*){0}'.format(layer)
        else:
            pattern_string = 'layer'
        pattern = re.compile(pattern_string, re.IGNORECASE)
            
        cortical_ids = self.descendant_ids(315)[0]
        return self.filter_nodes(lambda x: re.search(pattern, x['safe_name']) \
                                           and x['id'] in cortical_ids)
        
        
    def structure_descends_from(self, child_id, parent_id):
        return parent_id in self.ancestor_ids(child_id)
        
        
    @classmethod
    def from_ontologies_api(cls, oapi, structure_graph_id, 
                            insert_structure_sets=True):
                            
        structures = oapi.get_structures(structure_graph_id)
        
