# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

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

    def rename_columns(self,
                       data,
                       new_old_name_tuples=None):
        if new_old_name_tuples == None:
            new_old_name_tuples = []
            
        for new_name, old_name in new_old_name_tuples:
            data.columns = [new_name if c == old_name else c
                            for c in data.columns]                    
    
    
    def load_csv(self,
                 path,
                 rename=None,
                 index=None):
        # depend on external code to write this, just reload
        data = pd.DataFrame.from_csv(path)

        self.rename_columns(data, rename)

        if index is not None:        
            data.set_index([index], inplace=True)

        return data


    def load_json(self,
                  path,
                  rename=None,
                  index=None):
        data = pj.read_json(path, orient='records')

        self.rename_columns(data, rename)
        
        if index is not None:        
            data.set_index([index], inplace=True)

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