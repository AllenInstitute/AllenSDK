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
             'name': 'root', 'structure_sets':[{'id': 1}, {'id': 4}]}]


@pytest.fixture(scope='function')
def old_nodes():

    return [{'id': 0, 'structure_id_path': '/0/', 
             'color_hex_triplet': '000000', 'acronym': 'rt', 
             'name': 'root', 'parent_structure_id': 12}]


@pytest.fixture(scope='function')
def experiments():

    return [{'num-voxels': 100, 'injection-volume': 99, 'sum': 98, 
             'name': 'foo', 'transgenic-line': 'most_creish', 
             'structure-id': 97,}]


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


def test_init(mcc, fn_temp_dir):

    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    assert( os.path.exists(manifest_path) )


def test_get_annotation_volume(mcc, fn_temp_dir):

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'annotation', 'ccf_2016', 
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

    assert(obtained.node_ids()[0] == 0)
    assert( os.path.exists(path) )


def test_get_ontology(mcc, fn_temp_dir, old_nodes):

    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter('always')

        with mock.patch('allensdk.api.queries.ontologies_api.'
                        'OntologiesApi.model_query', 
                        return_value=old_nodes) as p:

            mcc.get_ontology()
            mcc.get_ontology()

            p.assert_called_once()
            assert(len(c) == 6)


def test_get_structures(mcc, fn_temp_dir, old_nodes):

    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter('always')

        with mock.patch('allensdk.api.queries.ontologies_api.'
                        'OntologiesApi.model_query', 
                        return_value=old_nodes) as p:

            obtained = mcc.get_structures()
            mcc.get_structures()

            p.assert_called_once()
            assert obtained['acronym'][0] == old_nodes[0]['acronym']
            assert len(c) == 2


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


def test_filter_experiments(mcc, fn_temp_dir, experiments):

    pass_line = mcc.filter_experiments(experiments, cre=True)
    fail_line = mcc.filter_experiments(experiments, cre=False)

    assert len(pass_line) == 1
    assert len(fail_line) == 0


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

    class FakeTree(object):
        def descendant_ids(self, list_of_things):
            return [list_of_things]
    mcc.get_structure_tree = lambda *a, **k: FakeTree()

    annot = np.arange(125).reshape((5, 5, 5))
    mcc.get_annotation_volume = lambda *a, **k: (annot, 'foo')

    path = os.path.join(fn_temp_dir, 'annotation', 'ccf_2016', 'structure_masks', 
                        'resolution_25', 'structure_{0}.nrrd'.format(12))

    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter('always')

        mask, _ = mcc.get_structure_mask(12)

        # also make sure we can do this for pd.Series input for backwards compatibility
        mask, _ = mcc.get_structure_mask(pd.Series([12]))

    assert( mask.sum() == 1 )
    #assert( len(c) == 2 )
    assert( os.path.exists(path) )

    with pytest.raises(ValueError):
        mask, _ = mcc.get_structure_mask("fish")


def test_make_structure_mask(mcc):

    annot = np.arange(125).reshape((5, 5, 5))
    sids = [0, 1, 2, 3, 4]

    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter('always')

        mask = mcc.make_structure_mask(sids, annot)

    #assert(len(c) == 1)
    assert mask.sum() == 5


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
        assert( out == map(int, inp) )
