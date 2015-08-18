#from allensdk.api.queries.structure.ontologies_api import OntologiesApi
import os
from allensdk.core import json_utilities as ju
import pandas as pd
from allensdk.config.manifest import Manifest


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

            
            self.manifest = Manifest(ju.read(file_name)['manifest'], os.path.dirname(file_name))
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
        

if __name__ == '__main__':
    c = Cache()
    c.load_manifest('manifest.json')
