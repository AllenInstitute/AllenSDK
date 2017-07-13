# Copyright 2016 Allen Institute for Brain Science
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


import os

import pandas as pd
import pandas.io.json as pj

import pytest
from mock import MagicMock

from allensdk.api.cache import Cache
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
