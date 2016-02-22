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

from allensdk.config.manifest import Manifest
import allensdk.core.json_utilities as ju
import pandas as pd
import pandas.io.json as pj
import os


class Cache(object):
    def __init__(self,
                 manifest=None,
                 cache=True):
        self.cache = cache
        self.load_manifest(manifest)
    

    def get_cache_path(self, file_name, manifest_key, *args):
        '''Helper method for accessing path specs from manifest keys.
        
        Parameters
        ----------
        file_name : string
        manifest_key : string
        args : ordered parameters
    
        Returns
        -------
        string or None
            path
        '''
        if self.cache:
            if file_name:
                return file_name
            elif self.manifest:
                return self.manifest.get_path(manifest_key, *args)

        return None
        

    def load_manifest(self, file_name):
        '''Read a keyed collection of path specifications.
        
        Parameters
        ----------
        file_name : string
            path to the manifest file
        
        Returns
        -------
        Manifest
        '''
        if file_name != None:
            if not os.path.exists(file_name):

                # make the directory if it doesn't exist already
                dirname = os.path.dirname(file_name)
                Manifest.safe_mkdir(dirname)

                self.build_manifest(file_name)

            
            self.manifest = Manifest(ju.read(file_name)['manifest'], os.path.dirname(file_name))
        else:
            self.manifest = None


    def build_manifest(self, file_name):
        '''Creation of default path speifications.
        
        Parameters
        ----------
        file_name : string
            where to save it
        '''
        raise Exception("This function must be defined in the appropriate subclass")


    def manifest_dataframe(self):
        '''Convenience method to view manifest as a pandas dataframe.
        '''
        return pd.DataFrame.from_dict(self.manifest.path_info,
                                      orient='index')

    def rename_columns(self,
                       data,
                       new_old_name_tuples=None):
        '''Convenience method to rename columns in a pandas dataframe.
        
        Parameters
        ----------
        data : dataframe
            edited in place.
        new_old_name_tuples : list of string tuples (new, old)
        '''
        if new_old_name_tuples == None:
            new_old_name_tuples = []
            
        for new_name, old_name in new_old_name_tuples:
            data.columns = [new_name if c == old_name else c
                            for c in data.columns]                    
    
    
    def load_csv(self,
                 path,
                 rename=None,
                 index=None):
        '''Read a csv file as a pandas dataframe.
        
        Parameters
        ----------
        rename : list of string tuples (new old), optional
            columns to rename
        index : string, optional
            post-rename column to use as the row label.
        '''
        data = pd.DataFrame.from_csv(path)

        self.rename_columns(data, rename)

        if index is not None:        
            data.set_index([index], inplace=True)

        return data


    def load_json(self,
                  path,
                  rename=None,
                  index=None):
        '''Read a json file as a pandas dataframe.
        
        Parameters
        ----------
        rename : list of string tuples (new old), optional
            columns to rename
        index : string, optional
            post-rename column to use as the row label.
        '''
        data = pj.read_json(path, orient='records')

        self.rename_columns(data, rename)
        
        if index is not None:        
            data.set_index([index], inplace=True)

        return data
    

    def wrap(self, fn, path, cache,
             save_as_json=True,
             return_dataframe=False,
             index=None,
             rename=None,
             **kwargs):
        '''make an rma query, save it and return the dataframe.
        
        Parameters
        ----------
        fn : function reference
            makes the actual query using kwargs.
        path : string
            where to save the data
        cache : boolean
            True will make the query, False just loads from disk
        save_as_json : boolean, optional
            True (default) will save data as json, False as csv
        return_dataframe : boolean, optional
            True will cast the return value to a pandas dataframe, False (default) will not 
        index : string, optional
            column to use as the pandas index
        rename : list of string tuples, optional
            (new, old) columns to rename
        kwargs : objects
            passed through to the query function
        
        Returns
        -------
        dict or DataFrame
            data type depends on return_dataframe option.
        
        Notes
        -----
        Column renaming happens after the file is reloaded for json
        '''
        if cache == True:
            json_data = fn(**kwargs)
            
            if save_as_json == True:
                ju.write(path, json_data)
            else:
                df = pd.DataFrame(json_data)
                self.rename_columns(df, rename)
                
                if index is not None:
                    df.set_index([index], inplace=True)
        
                df.to_csv(path)
    
        # read it back in
        if save_as_json == True:
            if return_dataframe == True:
                data = pj.read_json(path, orient='records')
                self.rename_columns(data, rename)
                if index != None:
                    data.set_index([index], inplace=True)
            else:
                data = ju.read(path)
        elif return_dataframe == True:
            data = pd.DataFrame.from_csv(path)
        else:
            raise ValueError('save_as_json=False cannot be used with return_dataframe=False')
        
        return data
