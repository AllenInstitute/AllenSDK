# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import os

import pandas as pd
import pandas.io.json as pj
import numpy as np

import pytest
from mock import MagicMock, mock_open, patch

from allensdk.api.cache import Cache, memoize
from allensdk.api.queries.rma_api import RmaApi
import allensdk.core.json_utilities as ju
from allensdk.config.manifest import ManifestVersionError
from allensdk.config.manifest_builder import ManifestBuilder

from allensdk.test_utilities.temp_dir import fn_temp_dir


@pytest.fixture
def cache():
    return Cache()


@pytest.fixture
def rma():
    return RmaApi()


@pytest.fixture
def wavefront_obj():
    return '''

v 8578 5484.96 5227.57
v 8509.2 5487.54 5237.07
v 8564.38 5522.13 5220.41
v 8631.93 5497.82 5228.33
v 8517.88 5542.95 5234.53
v 8615.26 5563.22 5224.48

# i'm a comment!

vn -0.0247061 -0.352726 -0.935401
vn -0.235489 -0.190095 -0.953105
vn -0.0880336 -0.0323767 -0.995591
vn 0.122706 -0.209891 -0.969994
vn -0.343738 0.217978 -0.913416
vn 0.0753706 0.16324 -0.983703

I should be a comment, but am not

f 1//1 2//2 3//3 
f 4//4 1//1 3//3 
f 3//3 2//2 5//5 
f 6//6 3//3 5//5 

    '''


def test_version_update(fn_temp_dir):

    class DummyCache(Cache):
        
        VERSION = None
    
        def build_manifest(self, file_name):
            manifest_builder = ManifestBuilder()
            manifest_builder.set_version(DummyCache.VERSION)
            manifest_builder.write_json_file(file_name)

    mpath = os.path.join(fn_temp_dir, 'manifest.json')
    dc = DummyCache(manifest=mpath)
    
    same_dc = DummyCache(manifest=mpath)
    
    with pytest.raises(ManifestVersionError):
        new_dc = DummyCache(manifest=mpath, version=1.0)


def test_wrap_json(rma, cache):
    msg = [{'whatever': True}]

    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': msg})
    ju.write = \
        MagicMock(name='write')

    ju.read = \
        MagicMock(name='read',
                  return_value=pd.DataFrame(msg))

    df = cache.wrap(rma.model_query,
                    'example.txt',
                    cache=True,
                    model='Hemisphere')

    assert df.loc[:, 'whatever'][0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    ju.write.assert_called_once_with('example.txt', msg)
    ju.read.assert_called_once_with('example.txt')


def test_wrap_dataframe(rma, cache):
    msg = [{'whatever': True}]

    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': msg})
    ju.write = \
        MagicMock(name='write')
    pj.read_json = \
        MagicMock(name='read_json',
                  return_value=msg)

    json_data = cache.wrap(rma.model_query,
                           'example.txt',
                           cache=True,
                           return_dataframe=True,
                           model='Hemisphere')

    assert json_data[0]['whatever']

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    ju.write.assert_called_once_with('example.txt', msg)
    pj.read_json.assert_called_once_with('example.txt', orient='records')

def test_memoize():

        import time

        @memoize
        def f(x):
            time.sleep(1)
            return x

        for ii in range(2):
            t0 = time.time()
            print(f(0), time.time() - t0)

        class FooBar(object):

            def __init__(self): pass

            @memoize
            def f(self, x):
                time.sleep(.1)
                return 1

        fb = FooBar()

        for ii in range(2):
            t0 = time.time()
            fb.f(0), time.time() - t0
