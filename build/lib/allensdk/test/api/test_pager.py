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
from . import SafeJsonMsg


@pytest.fixture
def pager():
    return RmaPager()

_msg = [{'whatever': True}]
_pd_msg = pd.DataFrame(_msg)
_csv_msg = pd.read_csv(StringIO.StringIO(""",whatever
0,True
"""), index_col=0)

_read_url_get_msg5 = [{'msg': _msg},
                      {'msg': _msg},
                      {'msg': _msg},
                      {'msg': _msg},
                      {'msg': _msg}]
_pj_msg5 = pd.DataFrame([{'whatever': True},
                         {'whatever': True},
                         {'whatever': True},
                         {'whatever': True},
                         {'whatever': True}])
_read_msg5 = [{'whatever': True},
              {'whatever': True},
              {'whatever': True},
              {'whatever': True},
              {'whatever': True}]


@pytest.fixture
def safe_read_url_get_msg5():
    return SafeJsonMsg(_read_url_get_msg5)

@pytest.fixture
def rma():
    return RmaApi()

@patch("allensdk.core.json_utilities.read_url_get",
       return_value={'msg': _msg})
def test_pageable_json(ju_read_url_get, rma):
    
    @pageable()
    def get_genes(**kwargs):
        return rma.model_query(model='Gene', **kwargs)

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
                     
    assert ju_read_url_get.call_args_list == list(expected_calls)


def test_all(safe_read_url_get_msg5, rma):
    with patch("allensdk.core.json_utilities.read_url_get", side_effect=safe_read_url_get_msg5) as ju_read_url_get:

        @pageable()
        def get_genes(**kwargs):
            return rma.model_query(model='Gene', **kwargs)

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
                        
        assert ju_read_url_get.call_args_list == list(expected_calls)


@pytest.mark.parametrize("cache_style",
                         (Cache.cache_csv,
                          Cache.cache_csv_json,
                          Cache.cache_csv_dataframe))
@patch("pandas.read_csv", return_value=_csv_msg)
@patch("os.makedirs")
def test_cacheable_pageable_csv(os_makedirs, read_csv,
                                cache_style, safe_read_url_get_msg5):
    with patch("allensdk.core.json_utilities.read_url_get", side_effect=safe_read_url_get_msg5) as ju_read_url_get:
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

        os_makedirs.assert_called_with('/path/to')

        base_query = ('http://api.brain-map.org/api/v2/data/query.json?'
                    'q=model::ApiCamCellMetric,'
                    'rma::options%5Bnum_rows$eq1%5D%5Bstart_row$eq{}%5D'
                    '%5Bcount$eqfalse%5D')

        expected_calls = map(lambda c: call(base_query.format(c)),
                            [0, 1, 2, 3, 4, 5])

        assert ju_read_url_get.call_args_list == list(expected_calls)
        read_csv.assert_called_once_with('/path/to/cam_cell_metrics.csv', parse_dates=True)

        assert csv_writerow.call_args_list == [call({'whatever': 'whatever'}),
                                            call({'whatever': True}),
                                            call({'whatever': True}),
                                            call({'whatever': True}),
                                            call({'whatever': True}),
                                            call({'whatever': True})]


@pytest.mark.parametrize("cache_style",
                         (Cache.cache_json,
                          Cache.cache_json_dataframe))
@patch("allensdk.core.json_utilities.read", return_value=_read_msg5)
@patch("pandas.io.json.read_json", return_value=_pj_msg5)
@patch("os.makedirs")
def test_cacheable_pageable_json(os_makedirs, pj_read_json,
                                 ju_read, cache_style, safe_read_url_get_msg5):
    with patch("allensdk.core.json_utilities.read_url_get", side_effect=safe_read_url_get_msg5) as ju_read_url_get:

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

        os_makedirs.assert_called_with('/path/to')

        base_query = \
            ('http://api.brain-map.org/api/v2/data/query.json?'
            'q=model::ApiCamCellMetric,'
            'rma::options%5Bnum_rows$eq1%5D%5Bstart_row$eq{}%5D'
            '%5Bcount$eqfalse%5D')

        expected_calls = map(lambda c: call(base_query.format(c)),
                            [0, 1, 2, 3, 4, 5])

        open_mock.assert_called_once_with('/path/to/cam_cell_metrics.json', 'wb')
        open_mock.return_value.write.assert_called_once_with('[\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  },\n  {\n    "whatever": true\n  }\n]')
        assert ju_read_url_get.call_args_list == list(expected_calls)
        assert len(cam_cell_metrics) == 5
