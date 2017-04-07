# Copyright 2016-2017 Allen Institute for Brain Science
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


import pytest
from mock import MagicMock, patch
from allensdk.api.cache import Cache, cacheable
from allensdk.api.queries.rma_api import RmaApi
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd
try:
    import StringIO
except:
    import io as StringIO
import os


_msg = [{'whatever': True}]
_pd_msg = pd.DataFrame(_msg)
_csv_msg = pd.DataFrame.from_csv(StringIO.StringIO(""",whatever
0,True
"""))

@pytest.fixture
def rma():
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': _msg})
    ju.write = \
        MagicMock(name='write')

    ju.read = \
        MagicMock(name='read',
                  return_value=_msg)

    pj.read_json = \
        MagicMock(name='read_json',
                  return_value=_pd_msg)

    pd.DataFrame.to_csv = \
        MagicMock(name='to_csv')

    pd.DataFrame.from_csv = \
        MagicMock(name='from_csv',
                  return_value=_csv_msg)
    
    os.makedirs = MagicMock(name='makedirs')

    return RmaApi()


def test_cacheable_csv_dataframe(rma):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.txt',
                         strategy='create',
                         **Cache.cache_csv_dataframe())

    assert df.loc[:, 'whatever'][0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    pd.DataFrame.to_csv.assert_called_once_with('/xyz/abc/example.txt')
    pd.DataFrame.from_csv.assert_called_once_with('/xyz/abc/example.txt')
    assert not ju.write.called, 'write should not have been called'
    assert not ju.read.called, 'read should not have been called'


def test_cacheable_json(rma):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.json',
                         strategy='create',
                         **Cache.cache_json())

    assert 'whatever' in df[0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not pd.DataFrame.to_csv.called, 'to_csv should not have been called'
    assert not pd.DataFrame.from_csv.called, 'from_csv should not have been called'
    ju.write.assert_called_once_with('/xyz/abc/example.json',
                                     _msg)
    ju.read.assert_called_once_with('/xyz/abc/example.json')


def test_excpt(rma):
    @cacheable()
    def get_hemispheres_excpt():
        return rma.model_query(model='Hemisphere',
                               excpt=['symbol'])

    df = get_hemispheres_excpt(path='/xyz/abc/example.json',
                         strategy='create',
                         **Cache.cache_json())

    assert 'whatever' in df[0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere,rma::options%5Bexcept$eqsymbol%5D')
    assert not pd.DataFrame.to_csv.called, 'to_csv should not have been called'
    assert not pd.DataFrame.from_csv.called, 'from_csv should not have been called'
    ju.write.assert_called_once_with('/xyz/abc/example.json',
                                     _msg)
    ju.read.assert_called_once_with('/xyz/abc/example.json')


def test_cacheable_no_cache_csv(rma):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.csv',
                         strategy='file',
                         **Cache.cache_csv())

    assert df.loc[:, 'whatever'][0]

    assert not ju.read_url_get.called
    assert not pd.DataFrame.to_csv.called, 'to_csv should not have been called'
    pd.DataFrame.from_csv.assert_called_once_with('/xyz/abc/example.csv')
    assert not ju.write.called, 'json write should not have been called'
    assert not ju.read.called, 'json read should not have been called'


def test_cacheable_json_dataframe(rma):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.json',
                         strategy='create',
                         **Cache.cache_json_dataframe())

    assert df.loc[:, 'whatever'][0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not pd.DataFrame.to_csv.called, 'to_csv should not have been called'
    assert not pd.DataFrame.from_csv.called, 'from_csv should not have been called'
    pj.read_json.assert_called_once_with('/xyz/abc/example.json',
                                         orient='records')
    ju.write.assert_called_once_with('/xyz/abc/example.json', _msg)
    assert not ju.read.called, 'json read should not have been called'


def test_cacheable_csv_json(rma):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/example.csv',
                         strategy='create',
                         **Cache.cache_csv_json())

    assert 'whatever' in df[0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    pd.DataFrame.to_csv.assert_called_once_with('/xyz/example.csv')
    pd.DataFrame.from_csv.assert_called_once_with('/xyz/example.csv')
    assert not pj.read_json.called, 'pj.read_json should not have been called'
    assert not ju.write.called, 'ju.write should not have been called'
    assert not ju.read.called, 'json read should not have been called'

# Not applicable any more because the parameter went away

def test_cacheable_no_save(rma, cache):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    data = get_hemispheres()

    assert 'whatever' in data[0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not pd.DataFrame.to_csv.called, 'to_csv should not have been called'
    assert not pd.DataFrame.from_csv.called, 'from_csv should not have been called'
    assert not ju.write.called, 'json write should not have been called'
    assert not ju.read.called, 'json read should not have been called'


def test_cacheable_no_save_dataframe(rma, cache):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    df = get_hemispheres(**Cache.nocache_dataframe())

    assert df.loc[:, 'whatever'][0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not pd.DataFrame.to_csv.called, 'to_csv should not have been called'
    assert not pd.DataFrame.from_csv.called, 'from_csv should not have been called'
    assert not ju.write.called, 'json write should not have been called'
    assert not ju.read.called, 'json read should not have been called'


def test_cacheable_lazy_csv_no_file(rma, cache):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    with patch('os.path.exists', MagicMock(return_value=False)) as ope:
        df = get_hemispheres(path='/xyz/abc/example.csv',
                             strategy='lazy',
                             **Cache.cache_csv())

    assert df.loc[:, 'whatever'][0]

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    pd.DataFrame.to_csv.assert_called_once_with('/xyz/abc/example.csv')
    pd.DataFrame.from_csv.assert_called_once_with('/xyz/abc/example.csv')
    assert not ju.write.called, 'json write should not have been called'
    assert not ju.read.called, 'json read should not have been called'


def test_cacheable_lazy_csv_file_exists(rma, cache):
    @cacheable()
    def get_hemispheres():
        return rma.model_query(model='Hemisphere')

    with patch('os.path.exists', MagicMock(return_value=True)) as ope:
        df = get_hemispheres(path='/xyz/abc/example.csv',
                             strategy='lazy',
                             **Cache.cache_csv())

    assert df.loc[:, 'whatever'][0]

    assert not ju.read_url_get.called
    assert not pd.DataFrame.to_csv.called
    pd.DataFrame.from_csv.assert_called_once_with('/xyz/abc/example.csv')
    assert not ju.write.called, 'json write should not have been called'
    assert not ju.read.called, 'json read should not have been called'