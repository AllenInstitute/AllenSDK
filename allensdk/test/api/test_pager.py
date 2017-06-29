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
from mock import MagicMock, call, patch, mock_open
from allensdk.api.queries.rma_pager import RmaPager, pageable
from allensdk.api.queries.rma_api import RmaApi
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd
from six.moves import builtins
import os
import simplejson as json
from allensdk.api.queries.rma_template import RmaTemplate
from allensdk.api.cache import cacheable, Cache
try:
    import StringIO
except:
    import io as StringIO


@pytest.fixture
def pager():
    return RmaPager()


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
    json.dumps = \
        MagicMock(name='dumps')

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

    return { 'ju_read_url_get': ju.read_url_get,
             'ju_write': ju.write,
             'ju_read': ju.read,
             'pj_read_json': pj.read_json,
             'to_csv': pd.DataFrame.to_csv,
             'from_csv': pd.DataFrame.from_csv,
             'os_makedirs': os.makedirs,
             'rma_api': RmaApi() }

@pytest.fixture
def rma5():
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  side_effect = [{'msg': _msg},
                                 {'msg': _msg},
                                 {'msg': _msg},
                                 {'msg': _msg},
                                 {'msg': _msg}])

    ju.read = \
        MagicMock(name='read',
                  return_value=[{'whatever': True},
                                {'whatever': True},
                                {'whatever': True},
                                {'whatever': True},
                                {'whatever': True}])

    pj.read_json = \
        MagicMock(name='read_json',
                  return_value=pd.DataFrame([{'whatever': True},
                                             {'whatever': True},
                                             {'whatever': True},
                                             {'whatever': True},
                                             {'whatever': True}]))

    pd.DataFrame.to_csv = \
        MagicMock(name='to_csv')

    pd.DataFrame.from_csv = \
        MagicMock(name='from_csv',
                  return_value=_csv_msg)
    
    os.makedirs = MagicMock(name='makedirs')

    return { 'ju_read_url_get': ju.read_url_get,
             #'ju_write': json.dumps,
             'ju_read': ju.read,
             'pj_read_json': pj.read_json,
             'to_csv': pd.DataFrame.to_csv,
             'from_csv': pd.DataFrame.from_csv,
             'os_makedirs': os.makedirs,
             'rma_api': RmaApi() }

def test_pageable_json(rma):
    @pageable()
    def get_genes(**kwargs):
        return rma['rma_api'].model_query(model='Gene',
                                          **kwargs)

    nr = 5
    pp = 1
    tr = nr*pp

    df = list(get_genes(num_rows=nr, total_rows=tr))

    assert df ==  [{'whatever': True},
                   {'whatever': True},
                   {'whatever': True},
                   {'whatever': True},
                   {'whatever': True}]

    base_query = \
        ('http://api.brain-map.org/api/v2/data/query.json?q=model::Gene'
         ',rma::options%5Bnum_rows$eq5%5D%5Bstart_row$eq{}%5D'
         '%5Bcount$eqfalse%5D')

    expected_calls = map(lambda c: call(base_query.format(c)),
                         [0, 1, 2, 3, 4])
                     
    assert rma['ju_read_url_get'].call_args_list == expected_calls


def test_all(rma5):
    @pageable()
    def get_genes(**kwargs):
        return rma5['rma_api'].model_query(model='Gene', **kwargs)

    nr = 1

    df = list(get_genes(num_rows=nr, total_rows='all'))

    assert df ==  [{'whatever': True},
                   {'whatever': True},
                   {'whatever': True},
                   {'whatever': True},
                   {'whatever': True}]

    base_query = \
        ('http://api.brain-map.org/api/v2/data/query.json?q=model::Gene'
         ',rma::options%5Bnum_rows$eq1%5D%5Bstart_row$eq{}%5D'
         '%5Bcount$eqfalse%5D')

    # we get one extra call if total_rows % num_rows == 0 with current implementation
    expected_calls = map(lambda c: call(base_query.format(c)),
                         [0, 1, 2, 3, 4, 5])
                     
    assert rma5['ju_read_url_get'].call_args_list == expected_calls


@pytest.mark.parametrize("cache_style",
                         (Cache.cache_csv,
                          Cache.cache_csv_json,
                          Cache.cache_csv_dataframe))
def test_cacheable_pageable_csv(rma5,
                                cache_style):
    archive_templates = \
        {"cam_cell_queries": [
            {'name': 'cam_cell_metric',
             'description': 'see name',
             'model': 'ApiCamCellMetric',
             'num_rows': 1000,
             'count': False
             } ] }

    rmat = RmaTemplate(query_manifest=archive_templates)

    @cacheable()
    @pageable(num_rows=2000)
    def get_cam_cell_metrics(*args,
                             **kwargs):
        return rmat.template_query("cam_cell_queries",
                                   'cam_cell_metric',
                                   *args,
                                   **kwargs)

    with patch(builtins.__name__ + '.open',
               mock_open(),
               create=True) as open_mock:
        with patch('csv.DictWriter.writerow') as csv_writerow:
            cam_cell_metrics = \
                get_cam_cell_metrics(strategy='create',
                                     path='/path/to/cam_cell_metrics.csv',
                                     num_rows=1,
                                     total_rows='all',
                                     **cache_style())

    rma5['os_makedirs'].assert_called_once_with('/path/to')

    base_query = \
        ('http://api.brain-map.org/api/v2/data/query.json?'
         'q=model::ApiCamCellMetric,'
         'rma::options%5Bnum_rows$eq1%5D%5Bstart_row$eq{}%5D'
         '%5Bcount$eqfalse%5D')

    expected_calls = map(lambda c: call(base_query.format(c)),
                         [0, 1, 2, 3, 4, 5])

    assert rma5['ju_read_url_get'].call_args_list == expected_calls
    assert csv_writerow.call_args_list == [call({'whatever': 'whatever'}),
                                           call({'whatever': True}),
                                           call({'whatever': True}),
                                           call({'whatever': True}),
                                           call({'whatever': True}),
                                           call({'whatever': True})]

@pytest.mark.parametrize("cache_style",
                         (Cache.cache_json,
                          Cache.cache_json_dataframe))
def test_cacheable_pageable_json(rma5,
                                 cache_style):
    archive_templates = \
        {"cam_cell_queries": [
            {'name': 'cam_cell_metric',
             'description': 'see name',
             'model': 'ApiCamCellMetric',
             'num_rows': 1000,
             'count': False
             } ] }

    rmat = RmaTemplate(query_manifest=archive_templates)

    @cacheable()
    @pageable(num_rows=2000)
    def get_cam_cell_metrics(*args,
                             **kwargs):
        return rmat.template_query("cam_cell_queries",
                                   'cam_cell_metric',
                                   *args,
                                   **kwargs)

    with patch(builtins.__name__ + '.open',
               mock_open(),
               create=True) as open_mock:
        open_mock.return_value.read = \
            MagicMock(name='read',
                      return_value=[{'whatever': True},
                                    {'whatever': True},
                                    {'whatever': True},
                                    {'whatever': True},
                                    {'whatever': True}])
        cam_cell_metrics = \
            get_cam_cell_metrics(strategy='create',
                                 path='/path/to/cam_cell_metrics.json',
                                 num_rows=1,
                                 total_rows='all',
                                 **cache_style())

    rma5['os_makedirs'].assert_called_once_with('/path/to')

    base_query = \
        ('http://api.brain-map.org/api/v2/data/query.json?'
         'q=model::ApiCamCellMetric,'
         'rma::options%5Bnum_rows$eq1%5D%5Bstart_row$eq{}%5D'
         '%5Bcount$eqfalse%5D')

    expected_calls = map(lambda c: call(base_query.format(c)),
                         [0, 1, 2, 3, 4, 5])

    open_mock.assert_called_once_with('/path/to/cam_cell_metrics.json', 'wb')
    open_mock.return_value.write.assert_called_once_with('[\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  }\n]')
    assert rma5['ju_read_url_get'].call_args_list == expected_calls
    assert len(cam_cell_metrics) == 5