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
import pytest
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
def unmocked_boc():
    boc = BrainObservatoryCache()
    
    return boc


@pytest.fixture
def brain_observatory_cache():
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
            # Download a list of all targeted areas
            boc = BrainObservatoryCache(manifest_file='boc/manifest.json',
                                        base_uri='http://api.brain-map.org')

    boc.api.json_msg_query = MagicMock(name='json_msg_query')

    return boc


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
def test_dataframe_query(brain_observatory_cache,
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


def test_dataframe_query_unmocked(unmocked_boc,
                                  example_filters,
                                  cells):
    brain_observatory_cache = unmocked_boc

    cells = brain_observatory_cache.get_cell_specimens(
        filters=example_filters)

    # total lines = 18260, can make fail by passing no filters
    #expected = 105
    assert len(cells) > 0 and len(cells) < 1000


def test_dataframe_query_between_unmocked(unmocked_boc,
                                          between_filter,
                                          cells):
    brain_observatory_cache = unmocked_boc

    cells = brain_observatory_cache.get_cell_specimens(
        filters=between_filter)

    # total lines = 18260, can make fail by passing no filters
    #expected = 15
    assert len(cells) > 0 and len (cells) < 1000


def test_dataframe_query_is_unmocked(unmocked_boc,
                                     cells):
    brain_observatory_cache = unmocked_boc

    is_filter = [
        {"field": "all_stim",
         "op": "is",
         "value": True }
    ]

    cells = brain_observatory_cache.get_cell_specimens(
        filters=is_filter)

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
