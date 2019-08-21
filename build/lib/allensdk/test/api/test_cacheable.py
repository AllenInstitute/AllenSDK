# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2016-2017. Allen Institute. All rights reserved.
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
import pytest
from mock import MagicMock, patch, mock_open
from allensdk.api.cache import Cache, cacheable
from allensdk.api.queries.rma_api import RmaApi
import pandas as pd
from six.moves import builtins
from allensdk.config.manifest import Manifest

try:
    import StringIO
except:
    import io as StringIO
import os


_msg = [{'whatever': True}]
_pd_msg = pd.DataFrame(_msg)
_csv_msg = pd.read_csv(StringIO.StringIO(""",whatever
0,True
"""), index_col=0)


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch('csv.DictWriter')
@patch('pandas.read_csv', return_value=_csv_msg)
def test_cacheable_csv_dataframe(read_csv, dictwriter, ju_read_url_get,
                                 ju_read, ju_write):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    with patch('allensdk.config.manifest.Manifest.safe_mkdir') as mkdir:
        with patch(builtins.__name__ + '.open',
                   mock_open(),
                   create=True) as open_mock:
            open_mock.return_value.write = MagicMock()
            df = get_hemispheres(path='/xyz/abc/example.txt',
                                 strategy='create',
                                 **Cache.cache_csv_dataframe())

    assert df.loc[:, 'whatever'][0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    read_csv.assert_called_once_with('/xyz/abc/example.txt', parse_dates=True)
    assert not ju_write.called, 'write should not have been called'
    assert not ju_read.called, 'read should not have been called'
    mkdir.assert_called_once_with('/xyz/abc')
    open_mock.assert_called_once_with('/xyz/abc/example.txt', 'w')


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch.object(Manifest, 'safe_mkdir')
@patch('pandas.read_csv', return_value=_csv_msg)
def test_cacheable_json(read_csv, mkdir, ju_read_url_get, ju_read, ju_write):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.json',
                         strategy='create',
                         **Cache.cache_json())

    assert 'whatever' in df[0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not read_csv.called, 'read_csv should not have been called'
    ju_write.assert_called_once_with('/xyz/abc/example.json',
                                                      _msg)
    ju_read.assert_called_once_with('/xyz/abc/example.json')


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch.object(Manifest, 'safe_mkdir')
def test_excpt(mkdir, ju_read_url_get, ju_read, ju_write):
    @cacheable()
    def get_hemispheres_excpt():
        return RmaApi().model_query(model='Hemisphere',
                                    excpt=['symbol'])

    df = get_hemispheres_excpt(path='/xyz/abc/example.json',
                               strategy='create',
                               **Cache.cache_json())

    assert 'whatever' in df[0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere,rma::options%5Bexcept$eqsymbol%5D')
    ju_write.assert_called_once_with('/xyz/abc/example.json', _msg)
    ju_read.assert_called_once_with('/xyz/abc/example.json')
    mkdir.assert_called_once_with('/xyz/abc')


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch('pandas.read_csv', return_value=_csv_msg)
def test_cacheable_no_cache_csv(read_csv, ju_read_url_get, ju_read, ju_write):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.csv',
                         strategy='file',
                         **Cache.cache_csv())

    assert df.loc[:, 'whatever'][0]

    assert not ju_read_url_get.called
    read_csv.assert_called_once_with('/xyz/abc/example.csv', parse_dates=True)
    assert not ju_write.called, 'json write should not have been called'
    assert not ju_read.called, 'json read should not have been called'


@patch("pandas.io.json.read_json", return_value=_pd_msg)
@patch("pandas.read_csv", return_value=_csv_msg)
@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch.object(Manifest, 'safe_mkdir')
def test_cacheable_json_dataframe(mkdir, ju_read_url_get, ju_read, ju_write,
                                  read_csv, mock_read_json):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    df = get_hemispheres(path='/xyz/abc/example.json',
                         strategy='create',
                         **Cache.cache_json_dataframe())

    assert df.loc[:, 'whatever'][0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not read_csv.called, 'read_csv should not have been called'
    mock_read_json.assert_called_once_with('/xyz/abc/example.json',
                                      orient='records')
    ju_write.assert_called_once_with('/xyz/abc/example.json', _msg)
    assert not ju_read.called, 'json read should not have been called'
    mkdir.assert_called_once_with('/xyz/abc')


@patch("pandas.io.json.read_json", return_value=_pd_msg)
@patch("pandas.read_csv", return_value=_csv_msg)
@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch('csv.DictWriter')
@patch.object(Manifest, 'safe_mkdir')
def test_cacheable_csv_json(mkdir, dictwriter, ju_read_url_get, ju_read,
                            ju_write, read_csv, mock_read_json):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    with patch(builtins.__name__ + '.open',
               mock_open(),
               create=True) as open_mock:
        open_mock.return_value.write = MagicMock()
        df = get_hemispheres(path='/xyz/example.csv',
                             strategy='create',
                             **Cache.cache_csv_json())

    assert 'whatever' in df[0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    read_csv.assert_called_once_with('/xyz/example.csv', parse_dates=True)
    dictwriter.return_value.writerow.assert_called()
    assert not mock_read_json.called, 'pj.read_json should not have been called'
    assert not ju_write.called, 'ju.write should not have been called'
    assert not ju_read.called, 'json read should not have been called'
    mkdir.assert_called_once_with('/xyz')
    open_mock.assert_called_once_with('/xyz/example.csv', 'w')


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch("pandas.read_csv")
@patch.object(pd.DataFrame, "to_csv")
def test_cacheable_no_save(to_csv, read_csv, ju_read_url_get, ju_read,
                           ju_write):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    data = get_hemispheres()

    assert 'whatever' in data[0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not to_csv.called, 'to_csv should not have been called'
    assert not read_csv.called, 'read_csv should not have been called'
    assert not ju_write.called, 'json write should not have been called'
    assert not ju_read.called, 'json read should not have been called'


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch("pandas.read_csv", return_value=_csv_msg)
@patch.object(pd.DataFrame, "to_csv")
def test_cacheable_no_save_dataframe(to_csv, read_csv, ju_read_url_get,
                                     ju_read, ju_write):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    df = get_hemispheres(**Cache.nocache_dataframe())

    assert df.loc[:, 'whatever'][0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    assert not to_csv.called, 'to_csv should not have been called'
    assert not read_csv.called, 'read_csv should not have been called'
    assert not ju_write.called, 'json write should not have been called'
    assert not ju_read.called, 'json read should not have been called'


@patch("pandas.read_csv", return_value=_csv_msg)
@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch('csv.DictWriter')
@patch.object(Manifest, 'safe_mkdir')
def test_cacheable_lazy_csv_no_file(mkdir, dictwriter, ju_read_url_get,
                                    ju_read, ju_write, read_csv):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    with patch('os.path.exists', MagicMock(return_value=False)) as ope:
        with patch(builtins.__name__ + '.open',
                   mock_open(),
                   create=True) as open_mock:
            open_mock.return_value.write = MagicMock()
            df = get_hemispheres(path='/xyz/abc/example.csv',
                                 strategy='lazy',
                                 **Cache.cache_csv())

    assert df.loc[:, 'whatever'][0]

    ju_read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    open_mock.assert_called_once_with('/xyz/abc/example.csv', 'w')
    dictwriter.return_value.writerow.assert_called()
    read_csv.assert_called_once_with('/xyz/abc/example.csv', parse_dates=True)
    assert not ju_write.called, 'json write should not have been called'
    assert not ju_read.called, 'json read should not have been called'


@patch("allensdk.core.json_utilities.write")
@patch("allensdk.core.json_utilities.read", return_value=_msg)
@patch("allensdk.core.json_utilities.read_url_get", return_value={'msg': _msg})
@patch("pandas.read_csv", return_value=_csv_msg)
def test_cacheable_lazy_csv_file_exists(read_csv, ju_read_url_get, ju_read,
                                        ju_write):
    @cacheable()
    def get_hemispheres():
        return RmaApi().model_query(model='Hemisphere')

    with patch('os.path.exists', MagicMock(return_value=True)) as ope:
        df = get_hemispheres(path='/xyz/abc/example.csv',
                             strategy='lazy',
                             **Cache.cache_csv())

    assert df.loc[:, 'whatever'][0]

    assert not ju_read_url_get.called
    read_csv.assert_called_once_with('/xyz/abc/example.csv', parse_dates=True)
    assert not ju_write.called, 'json write should not have been called'
    assert not ju_read.called, 'json read should not have been called'