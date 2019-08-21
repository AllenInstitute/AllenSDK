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
import SimpleITK as sitk


from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.structure_tree import StructureTree


@pytest.fixture
def cached_csv(tmpdir_factory):
    csv = str(tmpdir_factory.mktemp("cache_test").join("data.csv"))
    return csv


@pytest.fixture(scope='function')
def mcc(tmpdir_factory):
    manifest_file = tmpdir_factory.mktemp("mcc").join('manifest.json')
    return MouseConnectivityCache(manifest_file=str(manifest_file))


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
    return [{'data_set_id': 1, 'name': 'foo', 'storage_directory': 'meep', 'transgenic_line': { 'name': 'most_creish' },
             'injection_structures': '234/324', 'structure_id': 97},
            {'data_set_id': 2, 'name': 'bar', 'storage_directory': 'meep',  'transgenic_line': None,
             'injection_structures': '234/324/234', 'structure_id': 21}]


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


def test_init(mcc):
    assert( os.path.exists(mcc.manifest_path) )


def test_get_annotation_volume(mcc):

    eye = np.eye(100)
    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'annotation', 'ccf_2017', 
                        'annotation_25.nrrd')
 
    with mock.patch.object(mcc.api, "retrieve_file_over_http",
                           new=lambda a, b: nrrd.write(b, eye)):
        obtained, _ = mcc.get_annotation_volume()

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_annotation_volume()

    mock_rtrv.assert_not_called()
    assert( np.allclose(obtained, eye) ) 
    assert( os.path.exists(path) )


def test_get_template_volume(mcc):
    eye = np.eye(100)
    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'average_template_25.nrrd')

    with mock.patch.object(mcc.api, "retrieve_file_over_http",
                           new=lambda a, b: nrrd.write(b, eye)):
        obtained, _ = mcc.get_template_volume()

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_template_volume()

    mock_rtrv.assert_not_called()
    assert( np.allclose(obtained, eye) )
    assert( os.path.exists(path) )


def test_get_projection_density(mcc):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(os.path.dirname(mcc.manifest_path),
                        'experiment_{0}'.format(eid), 
                        'projection_density_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_projection_density(eid)

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_projection_density(eid)

    mock_rtrv.assert_not_called()
    assert( np.allclose(obtained, eye) )
    assert( os.path.exists(path) )


def test_get_injection_density(mcc):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'experiment_{0}'.format(eid), 
                        'injection_density_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_density(eid)

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_injection_density(eid)

    mock_rtrv.assert_not_called()
    assert( np.allclose(obtained, eye) )
    assert( os.path.exists(path) )


def test_get_injection_fraction(mcc):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(os.path.dirname(mcc.manifest_path),
                        'experiment_{0}'.format(eid), 
                        'injection_fraction_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_fraction(eid)

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_injection_fraction(eid)

    mock_rtrv.assert_not_called()
    assert( np.allclose(obtained, eye) )
    assert( os.path.exists(path) )


def test_get_data_mask(mcc):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'experiment_{0}'.format(eid), 
                        'data_mask_25.nrrd')

    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_data_mask(eid)

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_data_mask(eid)

    mock_rtrv.assert_not_called()
    assert( np.allclose(obtained, eye) )
    assert( os.path.exists(path) )


def test_get_structure_tree(mcc, new_nodes):

    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'structures.json')

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


def test_get_experiments(mcc, experiments):

    file_path = os.path.join(os.path.dirname(mcc.manifest_path), 'experiments.json')

    def new_fn(*args, **kwargs): return experiments
    
    with mock.patch.object(mcc.api, "model_query",
                           new=new_fn):
        obtained = mcc.get_experiments()

    with mock.patch.object(mcc.api, "model_query") as mock_squery:
        mcc.get_experiments()

    mock_squery.assert_not_called()
    assert os.path.exists(file_path)
    assert 'storage_directory' not in obtained[0]
    assert obtained[0]['transgenic_line'] == 'most_creish' 

    obtained = mcc.get_experiments(cre=['MOST_CREISH'])
    assert len(obtained) == 1


def test_filter_experiments(mcc, experiments):

    pass_line = mcc.filter_experiments(experiments, cre=True)
    fail_line = mcc.filter_experiments(experiments, cre=False)

    assert len(pass_line) == 1
    assert len(fail_line) == 1

    def fake_tree(*a, **k):
        class FakeTree(object):
            def descendant_ids(*a, **k):
                return [[97, 98], []]
        return FakeTree()

    with mock.patch.object(mcc, 'get_structure_tree', new=fake_tree) as p:
        sid_line = mcc.filter_experiments(experiments, cre=True, injection_structure_ids=[97, 98])

    assert len(sid_line) == 1

def test_rank_structures(mcc, top_injection_unionizes):

    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'experiment_{0}'.format(1), 
                        'structure_unionizes.csv')

    with mock.patch.object(mcc.api, "model_query",
                           lambda *args, **kwargs: top_injection_unionizes):
        obt = mcc.rank_structures([1], True, [15], [1, 2])

    assert(len(obt) == 1)
    exp = obt[0]
    assert(len(exp) == 1)
    st = exp[0]
    assert(st['structure_id'] == 15)
    assert(st['normalized_projection_volume'] == 0.25)


def test_default_structure_ids(mcc, new_nodes):

    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'structures.json')

    with mock.patch('allensdk.api.queries.ontologies_api.'
                    'OntologiesApi.model_query', 
                    return_value=new_nodes) as p:

        default_structure_ids = mcc.default_structure_ids
        assert(len(default_structure_ids) == 1)
        assert(default_structure_ids[0] == 0)


def test_get_experiment_structure_unionizes(mcc, unionizes):

    eid = 166218353
    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'experiment_{0}'.format(eid), 
                        'structure_unionizes.csv')

    with mock.patch.object(mcc.api, "model_query",
                           new=lambda *args, **kwargs: unionizes):
        obtained = mcc.get_experiment_structure_unionizes(eid)

    with mock.patch.object(mcc.api, "model_query") as mock_query:
        mcc.get_experiment_structure_unionizes(eid)

    mock_query.assert_not_called()
    assert obtained.loc[0, 'projection_intensity'] == 263.231
    assert os.path.exists(path)


def test_get_experiment_structure_unionizes_cache_roundtrip(mcc, unionizes,
                                                            cached_csv):

    eid = 166218353

    with mock.patch.object(mcc.api, "model_query",
                           new=lambda *args, **kwargs: unionizes):
        obtained = mcc.get_experiment_structure_unionizes(
            eid, file_name=cached_csv)
    pandas_data = pd.read_csv(cached_csv, index_col=0, parse_dates=True)

    assert obtained.loc[0, 'projection_intensity'] == 263.231
    assert(sorted(obtained.keys()) == sorted(pandas_data.columns))


def test_filter_structure_unionizes(mcc, unionizes):

    obtained = mcc.filter_structure_unionizes(pd.DataFrame(unionizes), 
                                              hemisphere_ids=[1])

    assert obtained.loc[0, 'volume'] == 0.016032

    obt_sid = mcc.filter_structure_unionizes(pd.DataFrame(unionizes),
                                              hemisphere_ids=[1],
                                              structure_ids=[1,60,90])

    assert obtained.loc[0, 'volume'] == 0.016032

def test_get_structure_unionizes(mcc, unionizes):

    with mock.patch.object(mcc, "get_experiment_structure_unionizes",
                           new=lambda *a, **k: pd.DataFrame(unionizes)):
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

    with mock.patch.object(mcc, "get_structure_unionizes",
                           new=lambda *a, **k: pd.DataFrame(unionizes)):
        class FakeTree(object):
            def value_map(*a, **k):
                return {1: 'one', 2: 'two'}
        with mock.patch.object(mcc, "get_structure_tree",
                               new=lambda *a, **k: FakeTree()):
            obtained = mcc.get_projection_matrix([1], [2], [1, 2], ['value'])

    assert np.allclose(obtained['matrix'], np.array([[30, 40]]))
    assert np.array_equal([ii['label'] for ii in obtained['columns']], 
                          ['two-L', 'two-R'])


def test_get_reference_space(mcc, new_nodes):

    tree = StructureTree(StructureTree.clean_structures(new_nodes))
    with mock.patch.object(mcc, "get_structure_tree",
                           new=lambda *a, **k: tree):
        annot = np.arange(125).reshape((5, 5, 5))
        with mock.patch.object(mcc, "get_annotation_volume",
                               new=lambda *a, **k: (annot, 'foo')):
            rsp_obt = mcc.get_reference_space()

    assert( np.allclose(rsp_obt.resolution, [25, 25, 25]) )
    assert( np.allclose( rsp_obt.annotation, annot ) )


def test_get_structure_mask(mcc):
  
    sid = 12

    eye = np.eye(100)
    path = os.path.join(os.path.dirname(mcc.manifest_path), 
                        'annotation', 'ccf_2017', 'structure_masks', 
                        'resolution_25', 'structure_{0}.nrrd'.format(sid))

    with mock.patch.object(mcc.api, "retrieve_file_over_http",
                           new=lambda a, b: nrrd.write(b, eye)):
        obtained, _ = mcc.get_structure_mask(sid)

    with mock.patch.object(mcc.api, "retrieve_file_over_http") as mock_rtrv:
        mcc.get_structure_mask(sid)

    mock_rtrv.assert_not_called()
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


def test_get_deformation_field(mcc):

    arr = np.random.rand(2, 4, 5, 3)

    def write_dfmfld(*a, **k):
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, str(k['header_path']), True) # TODO the str call here is only necessary in 2.7

    with mock.patch.object(mcc.api, 'download_deformation_field', new=write_dfmfld) as p:
        obtained = mcc.get_deformation_field(123)

    assert np.allclose(arr, obtained)


def test_get_affine_parameters(mcc):

    def new_fn(*args, **kwargs):
        return [{'alignment3d': {
            'trv_00': 1,
            'trv_01': 2,
            'trv_02': 3,
            'trv_03': 4,
            'trv_04': 5,
            'trv_05': 6,
            'trv_06': 7,
            'trv_07': 8,
            'trv_08': 9,
            'trv_09': 10,
            'trv_10': 11,
            'trv_11': 12,
        }}]
    
    expected = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    with mock.patch.object(mcc.api, "model_query", new=new_fn):
        obtained = mcc.get_affine_parameters(1245)

    assert np.allclose(expected, obtained)