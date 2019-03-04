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
import pytest
import os
import json
import pandas as pd
from zipfile import ZipFile
from mock import patch, mock_open, MagicMock
from test_brain_observatory_cache import CACHE_MANIFEST
from allensdk.core.brain_observatory_cache \
    import BrainObservatoryCache
from allensdk.api.queries.brain_observatory_api \
    import BrainObservatoryApi


try:
    import __builtin__ as builtins  # @UnresolvedImport
except:
    import builtins  # @UnresolvedImport

CELL_SPECIMEN_ZIP_URL = ("http://observatory.brain-map.org/visualcoding/"
                         "data/cell_metrics.csv.zip")


@pytest.fixture
def cells():
    return [{u'tld1_id': 177839004,
             u'natural_movie_two_small': None,
             u'natural_movie_one_a_small': None,
             u'speed_tuning_c_large': None,
             u'speed_tuning_c_small': None,
             u'drifting_grating_small': None,
             u'tld1_name': u'Cux2-CreERT2',
             u'imaging_depth': 275,
             u'tlr1_id': 265943423,
             u'pref_dir_dg': None,
             u'osi_sg': 0.728589701688166,
             u'osi_dg': None,
             u'tlr1_name': u'Ai93(TITL-GCaMP6f)',
             u'area': u'VISpm',
             u'pref_image_ns': 89.0,
             u'natural_movie_one_c_small': None,
             u'locally_sparse_noise_on_small': None,
             u'drifting_grating_large': None,
             u'experiment_container_id': 511498500,
             u'natural_movie_one_a_large': None,
             u'natural_movie_one_c_large': None,
             u'tld2_name': u'Camk2a-tTA',
             u'p_ns': 2.64407299505246e-05,
             u'natural_movie_three_large': None,
             u'pref_ori_sg': 30.0,
             u'speed_tuning_a_large': None,
             u'p_dg': None,
             u'time_to_peak_sg': 0.199499999999999,
             u'p_sg': 7.60972815250796e-05,
             u'time_to_peak_ns': 0.299249999999998,
             u'locally_sparse_noise_on_large': None,
             u'dsi_dg': None,
             u'pref_tf_dg': None,
             u'natural_movie_three_small': None,
             u'pref_sf_sg': 0.32,
             u'tld2_id': 177837320,
             u'locally_sparse_noise_off_large': None,
             u'locally_sparse_noise_off_small': None,
             u'cell_specimen_id': 517394843,
             u'pref_phase_sg': 0.5
            },
            {u'tld1_id': 177839004,
             u'natural_movie_two_small': None,
             u'natural_movie_one_a_small': None,
             u'speed_tuning_c_large': None,
             u'speed_tuning_c_small': None,
             u'drifting_grating_small': None,
             u'tld1_name': u'Cux2-CreERT2',
             u'imaging_depth': 275,
             u'tlr1_id': 265943423,
             u'natural_movie_two_large': None,
             u'speed_tuning_a_small': None,
             u'pref_dir_dg': None,
             u'osi_sg': 0.899272239777491,
             u'osi_dg': None,
             u'tlr1_name': u'Ai93(TITL-GCaMP6f)',
             u'area': u'VISpm',
             u'pref_image_ns': 15.0,
             u'natural_movie_one_c_small': None,
             u'locally_sparse_noise_on_small': None,
             u'drifting_grating_large': None,
             u'experiment_container_id': 511498500,
             u'natural_movie_one_a_large': None,
             u'natural_movie_one_c_large': None,
             u'tld2_name': u'Camk2a-tTA',
             u'p_ns': 0.000356823517642681,
             u'natural_movie_three_large': None,
             u'pref_ori_sg': 0.0,
             u'speed_tuning_a_large': None,
             u'p_dg': None,
             u'time_to_peak_sg': 0.565249999999996,
             u'p_sg': 0.0565790644804479,
             u'time_to_peak_ns': 0.432249999999997,
             u'locally_sparse_noise_on_large': None,
             u'dsi_dg': None,
             u'pref_tf_dg': None,
             u'natural_movie_three_small': None,
             u'pref_sf_sg': 0.32,
             u'tld2_id': 177837320,
             u'locally_sparse_noise_off_large': None,
             u'locally_sparse_noise_off_small': None,
             u'cell_specimen_id': 517394850,
             u'pref_phase_sg': 0.5}]


@pytest.fixture
def api():
    boi = BrainObservatoryApi()

    return boi


@pytest.fixture
def unmocked_boc(fn_temp_dir):
    manifest_file = os.path.join(fn_temp_dir, "unmocked_boc", "manifest.json")
    boc = BrainObservatoryCache(manifest_file=manifest_file)

    return boc


@pytest.fixture
def brain_observatory_cache(fn_temp_dir):
    boc = None

    try:
        manifest_data = bytes(CACHE_MANIFEST,
                              'UTF-8')  # Python 3
    except:
        manifest_data = bytes(CACHE_MANIFEST)  # Python 2.7

    with patch('os.path.exists',
               return_value=True):
        with patch(builtins.__name__ + ".open",
                   mock_open(read_data=manifest_data)):
            manifest_file = os.path.join(fn_temp_dir, "boc", "manifest.json")
            boc = BrainObservatoryCache(manifest_file=manifest_file,
                                        base_uri='http://api.brain-map.org')

    return boc


@pytest.fixture(scope="module")
def cell_specimen_table(tmpdir_factory):
    # download a zipped version of the cell specimen table for filter tests
    # as it is orders of magnitude faster
    api = BrainObservatoryApi()
    data_dir = str(tmpdir_factory.mktemp("data"))
    zipped = os.path.join("cell_specimens.zip")
    api.retrieve_file_over_http(CELL_SPECIMEN_ZIP_URL, zipped)
    df = pd.read_csv(ZipFile(zipped).open("cell_metrics.csv"),
                     true_values="t", false_values="f")
    js = json.loads(df.to_json(orient="records"))
    table_file = os.path.join(data_dir, "cell_specimens.json")
    with open(table_file, "w") as f:
        json.dump(js, f, indent=1)
    return table_file


@pytest.fixture
def example_filters():
    f = [{"field": "p_dg",
          "op": "<=",
          "value": 0.001 },
         {"field": "pref_dir_dg",
          "op": "=", "value": 45 },
         {"field": "area", "op": "in", "value": [ "VISpm" ] },
         {"field": "tld1_name",
          "op": "in",
          "value": [ "Rbp4-Cre", "Cux2-CreERT2", "Rorb-IRES2-Cre" ] }
    ]
    
    return f


@pytest.fixture
def between_filter():
    f = [{"field": "p_ns",
          "op": "between",
          "value": [ 0.00034, 0.00035 ] }
    ]
    
    return f


FILTER_OPERATORS = ["=", "<", ">", "<=", ">=", "between", "in", "is"]
QUERY_TEMPLATES = {
    "=": '({0} == {1})',
    "<": '({0} < {1})',
    ">": '({0} > {1})',
    "<=": '({0} <= {1})',
    ">=": '({0} >= {1})',
    "between": '({0} >= {1}) and ({0} <= {1})',
    "in": '({0} == {1})',
    "is": '({0} == {1})'
}


@pytest.mark.skipif(True, reason="not done")
@patch.object(BrainObservatoryApi, "json_msg_query")
def test_dataframe_query(mock_json_msg_query,
                         brain_observatory_cache,
                         between_filter,
                         cells):
    brain_observatory_cache = unmocked_boc
    with patch('os.path.exists',
               MagicMock(return_value=True)):
        with patch('allensdk.core.json_utilities.read',
                   MagicMock(return_value=cells)):
            cells = brain_observatory_cache.get_cell_specimens(
                filters=between_filter)

            assert len(cells) > 0


@pytest.mark.todo_flaky
def test_dataframe_query_unmocked(unmocked_boc,
                                  example_filters,
                                  cells,
                                  cell_specimen_table):
    brain_observatory_cache = unmocked_boc

    cells = brain_observatory_cache.get_cell_specimens(
        filters=example_filters,
        file_name=cell_specimen_table)

    # total lines = 18260, can make fail by passing no filters
    #expected = 105
    assert len(cells) > 0 and len(cells) < 1000


@pytest.mark.todo_flaky
def test_dataframe_query_between_unmocked(unmocked_boc,
                                          between_filter,
                                          cells,
                                          cell_specimen_table):
    brain_observatory_cache = unmocked_boc

    cells = brain_observatory_cache.get_cell_specimens(
        filters=between_filter,
        file_name=cell_specimen_table)

    # total lines = 18260, can make fail by passing no filters
    #expected = 15
    assert len(cells) > 0 and len (cells) < 1000


@pytest.mark.todo_flaky
def test_dataframe_query_is_unmocked(unmocked_boc,
                                     cells,
                                     cell_specimen_table):
    brain_observatory_cache = unmocked_boc

    is_filter = [
        {"field": "all_stim",
         "op": "is",
         "value": True }
    ]

    cells = brain_observatory_cache.get_cell_specimens(
        filters=is_filter,
        file_name=cell_specimen_table)

    assert len(cells) > 0


def test_dataframe_query_string_between(api):
    filters = [
        {"field": "p_ns",
         "op": "between",
         "value": [ 0.00034, 0.00035 ] }
    ]

    query_string = api.dataframe_query_string(filters)

    assert query_string == '(p_ns >= 0.00034) and (p_ns <= 0.00035)'


def test_dataframe_query_string_in(api):
    filters = [
        {"field": "name",
         "op": "in",
         "value": [ 'Abc', 'Def', 'Ghi' ] }
    ]

    query_string = api.dataframe_query_string(filters)

    assert query_string == "(name == ['Abc', 'Def', 'Ghi'])"


def test_dataframe_query_string_in_floats(api):
    filters = [
        {"field": "rating",
         "op": "in",
         "value": [ 9.9, 8.7, 0.1 ] }
    ]

    query_string = api.dataframe_query_string(filters)

    assert query_string == "(rating == [9.9, 8.7, 0.1])"


def test_dataframe_query_string_is_boolean(api):
    filters = [
        {"field": "fact_check",
         "op": "is",
         "value": False }
    ]

    query_string = api.dataframe_query_string(filters)

    assert query_string == "(fact_check == False)"


def test_dataframe_query_string_multi_filters(api,
                                              example_filters):
    query_string = api.dataframe_query_string(example_filters)

    assert query_string == ("(p_dg <= 0.001) & (pref_dir_dg == 45) & "
                            "(area == ['VISpm']) & " 
                            "(tld1_name == "
                            "['Rbp4-Cre', 'Cux2-CreERT2', 'Rorb-IRES2-Cre'])")
