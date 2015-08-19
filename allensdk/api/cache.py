from allensdk.core import json_utilities as ju
from allensdk.config.manifest import Manifest
import pandas as pd
import pandas.io.json as pj
import os


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

    @classmethod
    def wrap(self, fn, path, cache,
             save_as_json=False,
             index=None,
             rename=None,
             **kwargs):
        if cache == True:
            json_data = fn(**kwargs)
            
            if save_as_json == True:
                ju.write(path, json_data)
            else:            
                df = pd.DataFrame(json_data)
                
                if rename is not None:
                    for rename_entry in rename:
                        (new_name, old_name) = rename_entry
                        df.columns = [new_name if c == old_name else c
                                      for c in df.columns]                    
                
                if index is not None:        
                    df.set_index([index], inplace=True)
        
                df.to_csv(path)            
    
        # read it back in
        if save_as_json == True:
            df = pj.read_json(path, orient='records')
            df.set_index([index], inplace=True)
        else:
            df = pd.DataFrame.from_csv(path)
        
        return df