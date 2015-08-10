#from allensdk.api.queries.structure.ontologies_api import OntologiesApi
import os
from allensdk.core import json_utilities as ju
import pandas as pd
from allensdk.config.model.manifest import Manifest


class Cache(object):
    def __init__(self,
                 manifest=None,
                 cache=True):
        self.cache = cache
        
        self.load_manifest(manifest)
    
    def safe_mkdir(self, directory):
        try:
            os.makedirs(directory)
        except Exception, e:
            print e.message
            

    def get_cache_path(self, file_name, manifest_key, *args):
        if self.cache:
            if file_name:
                return file_name
            elif self.manifest:
                return self.manifest.get_path(manifest_key, *args)

        return None
        

    def load_manifest(self, file_name):
        if file_name != None:
            if not os.path.exists(file_name):
                self.build_manifest(file_name)

            self.manifest = Manifest(ju.read(file_name)['manifest'])
        else:
            self.manifest = None


    def build_manifest(self, file_name):
        raise Exception("This function must be defined in the appropriate subclass")


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
