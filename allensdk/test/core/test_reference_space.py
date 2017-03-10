# Copyright 2017 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import pytest
import mock
import numpy as np

from allensdk.core.reference_space import ReferenceSpace
from allensdk.core.structure_tree import StructureTree


@pytest.fixture
def rsp():

    tree = [{'id': 1, 'structure_id_path': [1]}, 
            {'id': 2, 'structure_id_path': [1, 2]}, 
            {'id': 3, 'structure_id_path': [1, 3]}, 
            {'id': 4, 'structure_id_path': [1, 2, 4]}, 
            {'id': 5, 'structure_id_path': [1, 2, 5]}, 
            {'id': 6, 'structure_id_path': [1, 2, 5, 6]}, 
            {'id': 7, 'structure_id_path': [1, 7]}]
            
    # leaves are 6, 4, 3
    # additionally annotate 2, 5 for realism :)
    annotation = np.zeros((10, 10, 10))
    annotation[4:8, 4:8, 4:8] = 2
    annotation[5:7, 5:7, 5:7] = 5
    annotation[5:7, 5:7, 5] = 6
    annotation[7, 7, 7] = 4
    annotation[8:10, 8:10, 8:10] = 3

    return ReferenceSpace(StructureTree(tree), annotation, [10, 10, 10])
    
    
def test_direct_voxel_counts(rsp):

    obt_one = rsp.direct_voxel_map
    obt_two = rsp.direct_voxel_map
    
    assert( obt_one[3] == 8 )
    assert( obt_one[2] == 4**3 - 2**3 - 1 )
    assert( obt_two[1] == 0 )
    assert( obt_two[2] == 4**3 - 2**3 - 1 )    

    
def test_total_voxel_counts(rsp):

    obt = rsp.total_voxel_map
    
    assert( obt[2] == 4**3 )
    assert( obt[6] == 4 )   
    
    
def test_remove_unassigned(rsp):

    rsp.remove_unassigned()
    node_ids = rsp.structure_tree.node_ids()
    
    assert( 1 in node_ids )
    assert( 7 not in node_ids )
    
    
def test_make_structure_mask(rsp):

    exp = np.zeros((10, 10, 10))
    exp[4:8, 4:8, 4:8] = 1
    exp[8:10, 8:10, 8:10] = 1
    obt = rsp.make_structure_mask([2, 3, 7])

    assert( np.allclose(obt, exp) )
    
    
def test_make_structure_mask_direct(rsp):

    exp = np.zeros((10, 10, 10))
    exp[5:7, 5:7, 6:7] = 1
    obt = rsp.make_structure_mask([5], True)

    assert( np.allclose(obt, exp) )
    
    
def test_many_structure_masks(rsp):

    cb = mock.MagicMock()
    
    [ii for ii in rsp.many_structure_masks([2, 3], output_cb=cb)]
    
    assert( cb.call_count == 2 )
    
    
def test_many_structure_masks_default_cb(rsp):
    
    rsp.make_structure_mask = mock.MagicMock(return_value=2)
    for item in rsp.many_structure_masks([1]):
        assert( np.allclose(item, [1, 2]) )
    
    
def test_check_coverage(rsp):
    
    mask = np.zeros((10, 10, 10))
    mask[7:10, 7:10, 7:10] = 1
    
    obt = rsp.check_coverage([3], mask)
    assert( np.count_nonzero(obt) == 27 - 8 )
    
    
def test_validate_structures(rsp):

    rsp.structure_tree.has_overlaps = mock.MagicMock()
    rsp.check_coverage = mock.MagicMock()
    
    rsp.validate_structures(1, 2)
    
    rsp.structure_tree.has_overlaps.assert_called_with(1)
    rsp.check_coverage.assert_called_with(1, 2)
    

def test_downsample(rsp):
    
    target = rsp.downsample((10, 20, 20))
    
    assert( np.allclose(target.annotation.shape, [10, 5, 5]) )
    
    
def test_get_slice_image(rsp):

    cmap = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [1, 2, 3], 
            4: [0, 0, 0], 5: [0, 0, 0], 6: [0, 0, 0], 7: [0, 0, 0], }
            
    image = rsp.get_slice_image(0, 90, cmap=cmap)
    
    assert( image[:, :, 0].sum() == 4 ) 
    
    
def test_direct_voxel_map_setter(rsp):
    
    rsp.direct_voxel_map = 4
    assert( rsp.direct_voxel_map == 4 )
    
    
def test_total_voxel_map_setter(rsp):
    
    rsp.total_voxel_map = 3
    assert( rsp.total_voxel_map == 3 )  
