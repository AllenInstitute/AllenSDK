import tempfile
import os
from functools import wraps
import scipy.sparse as sps
import numpy as np
import pandas as pd
import json

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {key:[item[key] for item in list_of_dicts] for key in list_of_dicts[0].keys() }

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

if __name__ == "__main__":

    import numpy as np

    # @memoize
    @cache(strategy='lazy')
    def f(x):
        return 5

    # print f
    print f(5)

    print list_of_dicts_to_dict_of_lists([{'a':1, 'b':0},{'a':5, 'b':7}])