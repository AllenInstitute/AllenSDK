import tempfile
import os
from functools import wraps
import scipy.sparse as sps
import numpy as np
import pandas as pd
import json

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {key:[item[key] for item in list_of_dicts] for key in list_of_dicts[0].keys() }

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key, **kwargs):
            return dict.__getitem__(self, (key, tuple(kwargs.items())))

        def __missing__(self, key):

            ret = self[key] = f(*key[0], **dict(key[1]))
            return ret

    return memodict().__getitem__

def cache(strategy='lazy', reader=None, writer=None, path=None):
    if path is None:
        def decor(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                if 'path' in kwargs:
                    path = kwargs.pop('path')
                    if strategy == 'lazy':
                        if os.path.exists(path):
                            return reader(path)
                        else:
                            data = f(*args, **kwargs)
                            writer(path, data)
                            return data
                else:
                    data = f(*args, **kwargs)
                    return data

            return wrapped
    else:
        def decor(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                if strategy == 'lazy':
                    if os.path.exists(path):
                        return reader(path)
                    else:
                        data = f(*args, **kwargs)
                        writer(path, data)
                        return data

            return wrapped
    return decor

def get_temp_file_name():
    f = tempfile.NamedTemporaryFile(delete=False)
    temp_file_name = f.name
    f.close()
    os.remove(f.name)
    return temp_file_name

def get_npy_reader_writer(mmap_mode='r'):

    def reader(path):
        return np.load(path, mmap_mode=mmap_mode)

    return {'reader':reader, 'writer':np.save}

def h5_cache(strategy='lazy', path=None):

    def decor(f):
        @wraps(f)
        def wrapped(*args):
            key = '/'.join(map(str, sorted(args)))

            reader_writer_dict = get_h5_reader_writer(key)
            reader = get_h5_reader_writer(key)['reader']
            writer = get_h5_reader_writer(key)['writer']

            if strategy == 'lazy':
                if os.path.exists(path) and key_in_h5_file(path, key):
                    return reader(path)
                else:
                    data = f(*args)
                    writer(path, data)
                    return data
            else:
                raise NotImplementedError


        return wrapped
    return decor

def get_cache_json_reader_writer(post=lambda x:x):

    def reader(path):
        data = json.load(open(path, 'r'))
        return post(data)

    writer = lambda path, data: json.dump(data, open(path, 'w'))

    return {
         'writer': writer,
         'reader': reader
    }


def get_cache_csv_dataframe_reader_writer(compression=None):
    return {
         'writer': lambda p, x : pd.DataFrame(x).to_csv(p, compression=compression),
         'reader' : pd.DataFrame.from_csv
    }

def dict_generator(indict, pre=None):

    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key] ):
                    yield d
            elif isinstance(value, list):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield indict

def read_h5_group(g):
    return_dict = {}
    if len(g.attrs) > 0:
        return_dict['attrs'] = dict(g.attrs)
    for key in g:
        if key == 'data':
            return_dict[key] = g[key].value
        else:
            return_dict[key] = read_h5_group(g[key])

    return return_dict

def read_h5_group(g):
    return_dict = {}
    if len(g.attrs) > 0:
        return_dict['attrs'] = dict(g.attrs)
    for key in g:
        if key == 'data':
            return_dict[key] = g[key].value
        else:
            return_dict[key] = read_h5_group(g[key])

    return return_dict

if __name__ == "__main__":

    import numpy as np

    # @memoize
    @cache(strategy='lazy')
    def f(x):
        return 5

    # print f
    print f(5)

    print list_of_dicts_to_dict_of_lists([{'a':1, 'b':0},{'a':5, 'b':7}])