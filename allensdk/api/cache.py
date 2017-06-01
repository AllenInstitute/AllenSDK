# Copyright 2015-2017 Allen Institute for Brain Science
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

from allensdk.config.manifest import Manifest, ManifestVersionError
import allensdk.core.json_utilities as ju
import pandas as pd
import pandas.io.json as pj
import functools
import os
import logging
from allensdk.deprecated import deprecated
import csv

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key, **kwargs):
            return dict.__getitem__(self, (key, tuple(kwargs.items())))

        def __missing__(self, key):

            ret = self[key] = f(*key[0], **dict(key[1]))
            return ret

    return memodict().__getitem__

class Cache(object):
    _log = logging.getLogger('allensdk.api.cache')

    def __init__(self,
                 manifest=None,
                 cache=True,
                 version=None):
        self.cache = cache
        self.load_manifest(manifest, version)

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

    def load_manifest(self, file_name, version=None):
        '''Read a keyed collection of path specifications.

        Parameters
        ----------
        file_name : string
            path to the manifest file

        Returns
        -------
        Manifest
        '''
        if file_name is not None:
            if not os.path.exists(file_name):

                # make the directory if it doesn't exist already
                dirname = os.path.dirname(file_name)
                if dirname:
                    Manifest.safe_mkdir(dirname)

                self.build_manifest(file_name)

            try:
                self.manifest = Manifest(
                    ju.read(file_name)['manifest'], 
                    os.path.dirname(file_name), 
                    version=version)
            except ManifestVersionError as e:
                raise ManifestVersionError(("Your manifest file (%s) is out of date" +
                                            " (version '%s' vs '%s').  Please remove this file" +
                                            " and it will be regenerated for you the next"
                                            " time you instantiate this class.") % (file_name, e.found_version, e.version),
                                           e.version, e.found_version)


        else:
            self.manifest = None

    def build_manifest(self, file_name):
        '''Creation of default path speifications.

        Parameters
        ----------
        file_name : string
            where to save it
        '''
        raise Exception(
            "This function must be defined in the appropriate subclass")

    def manifest_dataframe(self):
        '''Convenience method to view manifest as a pandas dataframe.
        '''
        return pd.DataFrame.from_dict(self.manifest.path_info,
                                      orient='index')

    @staticmethod
    def json_remove_keys(data, keys):
        for r in data:
            for key in keys:
                del r[key]

        return data

    @staticmethod
    def remove_keys(data, keys=None):
        ''' DataFrame version
        '''
        if keys is None:
            keys = []
        
        for key in keys:
            del data[key]

    @staticmethod
    def json_rename_columns(data,
                            new_old_name_tuples=None):
        '''Convenience method to rename columns in a pandas dataframe.

        Parameters
        ----------
        data : dataframe
            edited in place.
        new_old_name_tuples : list of string tuples (new, old)
        '''
        if new_old_name_tuples is None:
            new_old_name_tuples = []

        for new_name, old_name in new_old_name_tuples:
            for r in data:
                r[new_name] = r[old_name]
                del r[old_name]

    @staticmethod
    def rename_columns(data,
                       new_old_name_tuples=None):
        '''Convenience method to rename columns in a pandas dataframe.

        Parameters
        ----------
        data : dataframe
            edited in place.
        new_old_name_tuples : list of string tuples (new, old)
        '''
        if new_old_name_tuples is None:
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

        Cache.rename_columns(data, rename)

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

        Cache.rename_columns(data, rename)

        if index is not None:
            data.set_index([index], inplace=True)

        return data

    @staticmethod
    def cacher(fn,
               *args,
               **kwargs):
        '''make an rma query, save it and return the dataframe.
    
        Parameters
        ----------
        fn : function reference
            makes the actual query using kwargs.
        path : string
            where to save the data
        strategy : string or None, optional
            'create' always generates the data,
            'file' loads from disk,
            'lazy' queries the server if no file exists,
            None generates the data and bypasses all caching behavior
        pre : function
            df|json->df|json, takes one data argument and returns filtered version, None for pass-through
        post : function
            df|json->?, takes one data argument and returns Object
        reader : function, optional
            path -> data, default NOP
        writer : function, optional
            path, data -> None, default NOP
        kwargs : objects
            passed through to the query function

        Returns
        -------
        Object or None
            data type depends on fn, reader and/or post methods.
        '''
        path = kwargs.pop('path', None)
        strategy = kwargs.pop('strategy', None)
        pre = kwargs.pop('pre', lambda d: d)
        post = kwargs.pop('post', None)
        reader = kwargs.pop('reader', None)
        writer = kwargs.pop('writer', None)

        if strategy is None:
            if writer or path:
                strategy = 'lazy'
            else:
                strategy = 'pass_through'

        if not strategy in ['lazy', 'pass_through', 'file', 'create']:
            raise ValueError("Unknown query strategy: {}.".format(strategy))

        if 'lazy' == strategy:
            if os.path.exists(path):
                strategy = 'file'
            else:
                strategy = 'create'

        if strategy == 'pass_through':
                data = fn(*args, **kwargs)
        elif strategy in ['create']:
            Manifest.safe_make_parent_dirs(path)

            if writer:
                data = fn(*args, **kwargs)
                data = pre(data)
                writer(path, data)
            else:
                data = fn(*args, **kwargs)

        if reader:
            data = reader(path)

        # Note: don't provide post if fn or reader doesn't return data
        if post:
            data = post(data)
            return data

        try:
            data
            return data
        except:
            pass

        return

    @staticmethod
    def csv_writer(pth, gen):
        csv_writer = None
    
        first_row = True
        row_count = 1
    
        with open(pth, 'w') as output:
            for row in gen:
                if first_row:
                    field_names = map(str, row.keys())
                    csv_writer = csv.DictWriter(output,
                                                fieldnames=field_names,
                                                delimiter=',',
                                                quoting=csv.QUOTE_ALL)
                    csv_writer.writeheader()
                    first_row = False
                Cache._log.info('row: {}'.format(row_count))
                row_count = row_count + 1
                csv_writer.writerow(row)

    @staticmethod
    def cache_csv_json():
        return {
             'writer': Cache.csv_writer,
             'reader': lambda f: pd.DataFrame.from_csv(f).to_dict('records')
        }

    @staticmethod
    def cache_csv_dataframe():
        return {
             'writer': Cache.csv_writer,
             'reader' : pd.DataFrame.from_csv
        }

    @staticmethod
    def nocache_dataframe():
        return {
             'post': pd.DataFrame
        }

    @staticmethod
    def nocache_json():
        return {
        }

    @staticmethod
    def cache_json_dataframe():
        return {
             'writer': ju.write,
             'reader': lambda p: pj.read_json(p, orient='records')
        }

    @staticmethod
    def cache_json():
        return {
            'writer': ju.write,
            'reader' : ju.read
        }

    @staticmethod
    def cache_csv():
        return {
            'writer': Cache.csv_writer,
            'reader': pd.DataFrame.from_csv
        }

    @staticmethod
    def pathfinder(file_name_position,
                   secondary_file_name_position=None,
                   path_keyword=None):
        '''helper method to find path argument in legacy methods written
        prior to the @cacheable decorator.  Do not use for new @cacheable methods.
        
        Parameters
        ----------
        file_name_position : integer
            zero indexed position in the decorated method args where file path may be found.
        secondary_file_name_position : integer
            zero indexed position in the decorated method args where tha file path may be found.
        path_keyword : string
            kwarg that may have the file path.
        
        Notes
        -----
        This method is only intended to provide backward-compatibility for some
        methods that otherwise do not follow the path conventions of the @cacheable
        decorator.
        '''
        def pf(*args, **kwargs):
            file_name = None

            if path_keyword is not None and path_keyword in kwargs:
                file_name = kwargs[path_keyword]
            else:
                if file_name_position < len(args):
                    file_name = args[file_name_position]

                if (file_name is None and
                    secondary_file_name_position and 
                    secondary_file_name_position < len(args)):
                    file_name = args[secondary_file_name_position]

            return file_name
        return pf

    @deprecated()
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
        if cache is True:
            json_data = fn(**kwargs)

            if save_as_json is True:
                ju.write(path, json_data)
            else:
                df = pd.DataFrame(json_data)
                Cache.rename_columns(df, rename)

                if index is not None:
                    df.set_index([index], inplace=True)

                df.to_csv(path)

        # read it back in
        if save_as_json is True:
            if return_dataframe is True:
                data = pj.read_json(path, orient='records')
                Cache.rename_columns(data, rename)
                if index is not None:
                    data.set_index([index], inplace=True)
            else:
                data = ju.read(path)
        elif return_dataframe is True:
            data = pd.DataFrame.from_csv(path)
        else:
            raise ValueError(
                'save_as_json=False cannot be used with return_dataframe=False')
    
        return data


def cacheable(strategy=None,
              pre=None,
              writer=None,
              reader=None,
              post=None,
              pathfinder=None):
    '''decorator for rma queries, save it and return the dataframe.

    Parameters
    ----------
    fn : function reference
        makes the actual query using kwargs.
    path : string
        where to save the data
    strategy : string or None, optional
        'create' always gets the data from the source (server or generated),
        'file' loads from disk,
        'lazy' creates the data and saves to file if no file exists,
        None queries the server and bypasses all caching behavior
    pre : function
        df|json->df|json, takes one data argument and returns filtered version, None for pass-through
    post : function
        df|json->?, takes one data argument and returns Object
    reader : function, optional
        path -> data, default NOP
    writer : function, optional
        path, data -> None, default NOP
    kwargs : objects
        passed through to the query function

    Returns
    -------
    dict or DataFrame
        data type depends on dataframe option.

    Notes
    -----
    Column renaming happens after the file is reloaded for json
    '''
    def decor(func):
        decor.strategy=strategy
        decor.pre = pre
        decor.writer = writer
        decor.reader = reader
        decor.post = post
        decor.pathfinder = pathfinder

        @functools.wraps(func)
        def w(*args,
              **kwargs):
            if decor.pathfinder and not 'pathfinder' in kwargs:
                pathfinder = decor.pathfinder
            else:
                pathfinder = kwargs.pop('pathfinder', None)

            if pathfinder and not 'path' in kwargs:
                found_path = pathfinder(*args, **kwargs)
                
                if found_path:
                    kwargs['path'] = found_path
            if decor.strategy and not 'strategy' in kwargs:
                kwargs['strategy'] = decor.strategy
            if decor.pre and not 'pre' in kwargs:
                kwargs['pre'] = decor.pre
            if decor.writer and not 'writer' in kwargs:
                kwargs['writer'] = decor.writer
            if decor.reader and not 'reader' in kwargs:
                kwargs['reader'] = decor.reader
            if decor.post and not 'post in kwargs':
                kwargs['post'] = decor.post

            result = Cache.cacher(func,
                                  *args,
                                  **kwargs)
            return result
        
        return w
    return decor
