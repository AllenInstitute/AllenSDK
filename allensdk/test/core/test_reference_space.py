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
import os

import pytest
import mock
import numpy as np
import nrrd
import pandas as pd

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


@pytest.fixture
def itksnap_rsp():
    tree = [
        {'id': 1, 'rgb_triplet': [1, 2, 3], 'acronym': 'b', 'structure_id_path': [1]},
        {'id': 5000, 'rgb_triplet': [4, 5, 6], 'acronym': 'a', 'structure_id_path': [1, 5000]},
    ]

    annotation = np.zeros((10, 10, 10))
    annotation[:, :, :5] = 1
    annotation[:, :, 7:] = 5000

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


def test_export_itksnap_labels(itksnap_rsp):

    annot, labels = itksnap_rsp.export_itksnap_labels(id_type=np.uint8)

    exp = np.zeros((10, 10, 10))
    exp[:, :, :5] = 2
    exp[:, :, 7:] = 1

    assert set(np.unique(annot)) == set([0, 1, 2])
    assert np.array_equal(labels['LABEL'][:], ['a', 'b'])
    assert set(labels['IDX'].values) == set([1, 2])
    assert np.allclose(exp, annot)


def test_write_itksnap_labels(itksnap_rsp, tmpdir_factory):

    tmpdir = str(tmpdir_factory.mktemp('test_write_itksnap_labels'))
    annot_path = os.path.join(tmpdir, 'annot.nrrd')
    labels_path = os.path.join(tmpdir, 'labels.csv')

    itksnap_rsp.write_itksnap_labels(annot_path, labels_path, id_type=np.uint8)
    exp_annot, exp_labels = itksnap_rsp.export_itksnap_labels(id_type=np.uint8)

    obt_annot, _ = nrrd.read(annot_path)
    assert np.allclose(obt_annot, exp_annot)

    obt_labels = pd.read_csv(
        labels_path, 
        delim_whitespace=True, 
        names=['IDX', '-R-', '-G-', '-B-', '-A-', 'VIS', 'MSH', 'LABEL'], 
        index_col=False
    )
    pd.testing.assert_frame_equal(obt_labels, exp_labels, check_index_type=False)

    assert os.path.exists(labels_path)
    assert os.path.exists(annot_path)

