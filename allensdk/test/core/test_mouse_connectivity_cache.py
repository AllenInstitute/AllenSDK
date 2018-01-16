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
import os
import warnings
import mock
import pytest
import numpy as np
import nrrd
import pandas as pd


from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.structure_tree import StructureTree
from allensdk.test_utilities.temp_dir import fn_temp_dir


@pytest.fixture(scope='function')
def mcc(fn_temp_dir):

    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    return MouseConnectivityCache(manifest_file=manifest_path)


@pytest.fixture(scope='function')
def new_nodes():

    return [{'id': 0, 'structure_id_path': '/0/', 
             'color_hex_triplet': '000000', 'acronym': 'rt', 
             'name': 'root', 'structure_sets':[{'id': 1}, {'id': 4}, {'id': 167587189}] }]


@pytest.fixture(scope='function')
def old_nodes():

    return [{'id': 0, 'structure_id_path': '/0/', 
             'color_hex_triplet': '000000', 'acronym': 'rt', 
             'name': 'root', 'parent_structure_id': 12}]

@pytest.fixture(scope='function')
def experiments():

    return [{'num-voxels': 100, 'injection-volume': 99, 'sum': 98, 
             'name': 'foo', 'transgenic-line': 'most_creish', 
             'structure-id': 97}]


@pytest.fixture(scope='function')
def unionizes():

    # note that I've mucked around with these values a bit
    return [{"hemisphere_id": 1, "id": 169991412, "is_injection": False,
             "max_voxel_density": 0.284863, "max_voxel_x": 7700, 
             "max_voxel_y": 6500, "max_voxel_z": 5000, 
             "normalized_projection_volume": 0.0, 
             "projection_density": 0.116754, "projection_energy": 30.7332,
             "projection_intensity": 263.231, "projection_volume": 0.0018718,
             "section_data_set_id": 166218353, "structure_id": 1,
             "sum_pixel_intensity": 99234900.0, "sum_pixels": 1308740.0,
             "sum_projection_pixel_intensity": 40221700.0,
             "sum_projection_pixels": 152800.0,
             "volume": 0.016032},
            {"hemisphere_id": 2, "id": 169991601, "is_injection": False,
             "max_voxel_density": 0.0614783, "max_voxel_x": 7500,
             "max_voxel_y": 4900, "max_voxel_z": 1700,
             "normalized_projection_volume": 0.0, 
             "projection_density": 0.0168009,
             "projection_energy": 1.96084, "projection_intensity": 116.71,
             "projection_volume": 0.00148144, 
             "section_data_set_id": 166218353, "structure_id": 60,
             "sum_pixel_intensity": 261941000.0, "sum_pixels": 7198050.0,
             "sum_projection_pixel_intensity": 14114200.0,
             "sum_projection_pixels": 120934.0, "volume": 0.0881761}]


@pytest.fixture(scope='function')
def top_injection_unionizes():
    return pd.DataFrame([{'experiment_id': 1, 'is_injection': True, 'hemisphere_id': 1, 'structure_id': 10, 'normalized_projection_volume': 0.75}, 
                         {'experiment_id': 1, 'is_injection': True, 'hemisphere_id': 2, 'structure_id': 15, 'normalized_projection_volume': 0.25}, 
                         {'experiment_id': 1, 'is_injection': False, 'hemisphere_id': 1, 'structure_id': 10, 'normalized_projection_volume': 2.0}, 
                         {'experiment_id': 1, 'is_injection': False, 'hemisphere_id': 2, 'structure_id': 11, 'normalized_projection_volume': 0.001}])


def test_init(mcc, fn_temp_dir):

    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    assert( os.path.exists(manifest_path) )


def test_get_annotation_volume(mcc, fn_temp_dir):

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'annotation', 'ccf_2017', 
                        'annotation_25.nrrd')

    mcc.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = mcc.get_annotation_volume()

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_annotation_volume()

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) ) 
    assert( os.path.exists(path) )


def test_get_template_volume(mcc, fn_temp_dir):

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'average_template_25.nrrd')

    mcc.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = mcc.get_template_volume()

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_template_volume()

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )


def test_get_projection_density(mcc, fn_temp_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid), 
                        'projection_density_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_projection_density(eid)

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_projection_density(eid)

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )


def test_get_injection_density(mcc, fn_temp_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid), 
                        'injection_density_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_density(eid)

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_injection_density(eid)

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )


def test_get_injection_fraction(mcc, fn_temp_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid), 
                        'injection_fraction_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_fraction(eid)

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_injection_fraction(eid)

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )


def test_get_data_mask(mcc, fn_temp_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid), 
                        'data_mask_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_data_mask(eid)

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_data_mask(eid)

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )


def test_get_structure_tree(mcc, fn_temp_dir, new_nodes):

    path = os.path.join(fn_temp_dir, 'structures.json')

    with mock.patch('allensdk.api.queries.ontologies_api.'
                    'OntologiesApi.model_query', 
                    return_value=new_nodes) as p:

        obtained = mcc.get_structure_tree()

        mcc.get_structure_tree()
        p.assert_called_once()

    assert( obtained.node_ids()[0] == 0 )
    
    cm_obt = obtained.get_colormap()
    assert(len(cm_obt[0]) == 3)

    assert( os.path.exists(path) )


def test_get_experiments(mcc, fn_temp_dir, experiments):

    file_path = os.path.join(fn_temp_dir, 'experiments.json')

    mcc.api.service_query = lambda a, parameters: experiments    
    obtained = mcc.get_experiments()

    mcc.api.service_query = mock.MagicMock()
    mcc.get_experiments()

    mcc.api.service_query.assert_not_called()
    assert os.path.exists(file_path)
    assert 'num_voxels' not in obtained[0]
    assert obtained[0]['transgenic-line'] == 'most_creish' 

    obtained = mcc.get_experiments(cre=['MOST_CREISH'])
    assert len(obtained) == 1


def test_filter_experiments(mcc, fn_temp_dir, experiments):

    pass_line = mcc.filter_experiments(experiments, cre=True)
    fail_line = mcc.filter_experiments(experiments, cre=False)

    assert len(pass_line) == 1
    assert len(fail_line) == 0

    sid_line = mcc.filter_experiments(experiments, cre=True,
                                      injection_structure_ids=[97,98])

    assert len(sid_line) == 1

def test_rank_structures(mcc, top_injection_unionizes, fn_temp_dir):

    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(1), 
                        'structure_unionizes.csv')

    mcc.api.model_query = lambda *args, **kwargs: top_injection_unionizes

    obt = mcc.rank_structures([1], True, [15], [1, 2])

    assert(len(obt) == 1)
    exp = obt[0]
    assert(len(exp) == 1)
    st = exp[0]
    assert(st['structure_id'] == 15)
    assert(st['normalized_projection_volume'] == 0.25)


def test_default_structure_ids(mcc, fn_temp_dir, new_nodes):

    path = os.path.join(fn_temp_dir, 'structures.json')

    with mock.patch('allensdk.api.queries.ontologies_api.'
                    'OntologiesApi.model_query', 
                    return_value=new_nodes) as p:

        default_structure_ids = mcc.default_structure_ids
        assert(len(default_structure_ids) == 1)
        assert(default_structure_ids[0] == 0)


def test_get_experiment_structure_unionizes(mcc, fn_temp_dir, unionizes):

    eid = 166218353
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid), 
                        'structure_unionizes.csv')

    mcc.api.model_query = lambda *args, **kwargs: unionizes
    obtained = mcc.get_experiment_structure_unionizes(eid)

    mcc.api.model_query = mock.MagicMock()
    mcc.get_experiment_structure_unionizes(eid)

    mcc.api.model_query.assert_not_called()
    assert obtained.loc[0, 'projection_intensity'] == 263.231
    assert os.path.exists(path)


def test_filter_structure_unionizes(mcc, unionizes):

    obtained = mcc.filter_structure_unionizes(pd.DataFrame(unionizes), 
                                              hemisphere_ids=[1])

    assert obtained.loc[0, 'volume'] == 0.016032

    obt_sid = mcc.filter_structure_unionizes(pd.DataFrame(unionizes),
                                              hemisphere_ids=[1],
                                              structure_ids=[1,60,90])

    assert obtained.loc[0, 'volume'] == 0.016032

def test_get_structure_unionizes(mcc, unionizes):

    mcc.get_experiment_structure_unionizes = \
        lambda *a, **k: pd.DataFrame(unionizes)
    obtained = mcc.get_structure_unionizes([1, 2, 3])

    assert obtained.shape[0] == 6


def test_get_projection_matrix(mcc):
    # yup

    unionizes = [{'experiment_id': 1, 
                  'structure_id': 2, 
                  'hemisphere_id': 1, 
                  'value': 30},
                 {'experiment_id': 1, 
                  'structure_id': 2, 
                  'hemisphere_id': 2, 
                  'value': 40},]

    mcc.get_structure_unionizes = lambda *a, **k: pd.DataFrame(unionizes)

    class FakeTree(object):
        def value_map(*a, **k):
            return {1: 'one', 2: 'two'}
    mcc.get_structure_tree = lambda *a, **k: FakeTree()

    obtained = mcc.get_projection_matrix([1], [2], [1, 2], ['value'])

    assert np.allclose(obtained['matrix'], np.array([[30, 40]]))
    assert np.array_equal([ii['label'] for ii in obtained['columns']], 
                          ['two-L', 'two-R'])


def test_get_reference_space(mcc, new_nodes):

    tree = StructureTree(StructureTree.clean_structures(new_nodes))
    mcc.get_structure_tree = lambda *a, **k: tree

    annot = np.arange(125).reshape((5, 5, 5))
    mcc.get_annotation_volume = lambda *a, **k: (annot, 'foo')

    rsp_obt = mcc.get_reference_space()

    assert( np.allclose(rsp_obt.resolution, [25, 25, 25]) )
    assert( np.allclose( rsp_obt.annotation, annot ) ) 


def test_get_structure_mask(mcc, fn_temp_dir):
  
    sid = 12

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'annotation', 'ccf_2017', 'structure_masks', 
                        'resolution_25', 'structure_{0}.nrrd'.format(sid))

    mcc.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = mcc.get_structure_mask(sid)

    mcc.api.retrieve_file_over_http = mock.MagicMock()
    mcc.get_structure_mask(sid)

    mcc.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) ) 
    assert( os.path.exists(path) )


@pytest.mark.parametrize('inp,fails', [(1, False), 
                                        (pd.Series([2]), False), 
                                        ('qwerty', True)])
def test_validate_structure_id(inp, fails):

    if fails:
        with pytest.raises(ValueError) as exc:
            MouseConnectivityCache.validate_structure_id(inp)
    else:
        out = MouseConnectivityCache.validate_structure_id(inp)
        assert( out == int(inp) )


@pytest.mark.parametrize('inp,fails', [([1, 2, 3], False), 
                                        ([pd.Series([2]), pd.Series([3])], False), 
                                        (['qwerty', 1], True)])
def test_validate_structure_ids(inp, fails):

    if fails:
        with pytest.raises(ValueError) as exc:
            MouseConnectivityCache.validate_structure_ids(inp)
    else:
        out = MouseConnectivityCache.validate_structure_ids(inp)
        assert( out == [ int(i) for i in inp ] )
