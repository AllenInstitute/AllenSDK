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
from mock import MagicMock, patch, call
from allensdk.api.queries.rma_pager import RmaPager, pageable
from allensdk.api.queries.rma_api import RmaApi
from allensdk.api.queries.rma_template import RmaTemplate
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd
import StringIO
import os


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

@pytest.fixture
def rma5():
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  side_effect = [{'msg': _msg},
                                 {'msg': _msg},
                                 {'msg': _msg},
                                 {'msg': _msg},
                                 {'msg': _msg}])
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


def test_pageable_json(rma, cache):
    @pageable()
    def get_genes(**kwargs):
        return rma.model_query(model='Gene',
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
                     
    assert ju.read_url_get.call_args_list == expected_calls


def test_all(rma5, cache):
    @pageable()
    def get_genes(**kwargs):
        return rma5.model_query(model='Gene', **kwargs)

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
                     
    assert ju.read_url_get.call_args_list == expected_calls
