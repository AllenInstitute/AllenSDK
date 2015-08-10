#from allensdk.api.queries.structure.ontologies_api import OntologiesApi
from allensdk.core import json_utilities as ju
import pandas as pd
from allensdk.config.model.manifest import Manifest


class Cache(object):
    def __init__(self,
                 manifest=None,
                 cache=True):
        self.cache = cache
        
        if manifest != None:
            self.manifest = Manifest(ju.read(manifest)['manifest'])
        else:
            self.manifest = None


    def manifest_dataframe(self):
        return pd.DataFrame.from_dict(self.manifest.path_info,
                                      orient='index')
        

    def load_csv(self, path):
        # depend on external code to write this, just reload
        data = pd.DataFrame.from_csv(path)            

        return data


#     def load_summary_structures(self):
#         path = self.manifest.get_path('SUMMARY_STRUCTURES')
#         
#         if self.cache == True:
#             # TODO: move to API, remove 1009 using pandas
#             # fetch "summary" structures (structure set id = 167587189)
#             summary_structures = \
#                 self.rma.model_query(
#                     'Structure',
#                     criteria="structure_sets[name$eq'Mouse Connectivity - Summary']",
#                     order=[self.rma.quote_string('structures.graph_order$asc')],
#                     num_rows='all',
#                     count=False)
#                 
#             print("%d structures retrieved." % (len(summary_structures)))
#             
#             df = pd.DataFrame(summary_structures)
#             df.set_index(['id'], inplace=True)                
#             df.to_csv(path)
# 
#         summary_structures = pd.DataFrame.from_csv(path)
#     
#         return summary_structures




    def load_primary_injection_structures(self):
        path = self.manifest.get_path('PRIMARY_INJECTION_STRUCTURES')
        
        primary_injection_structures = pd.DataFrame.from_csv(path)
    
        return primary_injection_structures


    def load_secondary_injection_structures(self):
        path = self.manifest.get_path('SECONDARY_INJECTION_STRUCTURES')
        
        # depend on external code to write this, just reload
        secondary_injection_structures = pd.DataFrame.from_csv(path)            
    
        return secondary_injection_structures
        

if __name__ == '__main__':
    c = Cache()
    c.load_manifest()