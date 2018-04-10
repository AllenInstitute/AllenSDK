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

import pytest
import mock
import numpy as np
import nrrd
import pandas as pd

from allensdk.core.reference_space_cache import ReferenceSpaceCache
from allensdk.core.structure_tree import StructureTree


@pytest.fixture()
def rsp_version():
    return 'annotation/look_a_version'


@pytest.fixture()
def resolution():
    return 25


@pytest.fixture(scope='function')
def old_nodes():

    return [{'id': 0, 'structure_id_path': '/0/', 
             'color_hex_triplet': '000000', 'acronym': 'rt', 
             'name': 'root', 'parent_structure_id': 12}]


@pytest.fixture(scope='function')
def new_nodes():

    return [{'id': 0, 'structure_id_path': '/0/', 
             'color_hex_triplet': '000000', 'acronym': 'rt', 
             'name': 'root', 'structure_sets':[{'id': 1}, {'id': 4}, {'id': 167587189}] }]


@pytest.fixture(scope='function')
def rsp(fn_temp_dir, rsp_version, resolution):

    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    return ReferenceSpaceCache(reference_space_key=rsp_version,
                               resolution=resolution, 
                               manifest=manifest_path)



def test_init(rsp, fn_temp_dir):

    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    assert( os.path.exists(manifest_path) )


def test_get_annotation_volume(rsp, fn_temp_dir, rsp_version, resolution):

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, rsp_version, 'annotation_{0}.nrrd'.format(resolution))

    rsp.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = rsp.get_annotation_volume()

    rsp.api.retrieve_file_over_http = mock.MagicMock()
    rsp.get_annotation_volume()

    rsp.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) ) 
    assert( os.path.exists(path) )


def test_get_template_volume(rsp, fn_temp_dir, resolution):

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'average_template_{0}.nrrd'.format(resolution))

    rsp.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = rsp.get_template_volume()

    rsp.api.retrieve_file_over_http = mock.MagicMock()
    rsp.get_template_volume()

    rsp.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )


def test_get_structure_tree(rsp, fn_temp_dir, new_nodes):

    path = os.path.join(fn_temp_dir, 'structures.json')

    with mock.patch('allensdk.api.queries.ontologies_api.'
                    'OntologiesApi.model_query', 
                    return_value=new_nodes) as p:

        obtained = rsp.get_structure_tree()

        rsp.get_structure_tree()
        p.assert_called_once()

    assert(obtained.node_ids()[0] == 0)
    
    cm_obt = obtained.get_colormap()
    assert(len(cm_obt[0]) == 3)

    assert( os.path.exists(path) )


def test_get_reference_space(rsp, new_nodes):

    tree = StructureTree(StructureTree.clean_structures(new_nodes))
    rsp.get_structure_tree = lambda *a, **k: tree

    annot = np.arange(125).reshape((5, 5, 5))
    rsp.get_annotation_volume = lambda *a, **k: (annot, 'foo')

    rsp_obt = rsp.get_reference_space()

    assert( np.allclose(rsp_obt.resolution, [25, 25, 25]) )
    assert( np.allclose( rsp_obt.annotation, annot ) ) 


def test_get_structure_mask(rsp, fn_temp_dir, rsp_version):
  
    sid = 12

    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, rsp_version, 'structure_masks', 
                        'resolution_25', 'structure_{0}.nrrd'.format(sid))

    rsp.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = rsp.get_structure_mask(sid)

    rsp.api.retrieve_file_over_http = mock.MagicMock()
    rsp.get_structure_mask(sid)

    rsp.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained, eye) ) 
    assert( os.path.exists(path) )


def test_get_structure_mesh(rsp, fn_temp_dir, rsp_version):
  
    sid = 12

    path = os.path.join(fn_temp_dir, rsp_version, 'structure_meshes','structure_{0}.obj'.format(sid))

    def write_obj(path):
        with open(path, 'w') as fil:
          fil.write('vn 1 2 4')

    expected = [1, 2, 4]

    rsp.api.retrieve_file_over_http = lambda a, b: write_obj(b)
    obtained = rsp.get_structure_mesh(sid)

    rsp.api.retrieve_file_over_http = mock.MagicMock()
    rsp.get_structure_mesh(sid)

    rsp.api.retrieve_file_over_http.assert_not_called()
    assert( np.allclose(obtained[1], expected) ) 
    assert( os.path.exists(path) )


@pytest.mark.parametrize('inp,fails', [(1, False), 
                                        (pd.Series([2]), False), 
                                        ('qwerty', True)])
def test_validate_structure_id(inp, fails):

    if fails:
        with pytest.raises(ValueError) as exc:
            ReferenceSpaceCache.validate_structure_id(inp)
    else:
        out = ReferenceSpaceCache.validate_structure_id(inp)
        assert( out == int(inp) )


@pytest.mark.parametrize('inp,fails', [([1, 2, 3], False), 
                                        ([pd.Series([2]), pd.Series([3])], False), 
                                        (['qwerty', 1], True)])
def test_validate_structure_ids(inp, fails):

    if fails:
        with pytest.raises(ValueError) as exc:
            ReferenceSpaceCache.validate_structure_ids(inp)
    else:
        out = ReferenceSpaceCache.validate_structure_ids(inp)
        assert( out == list(map(int, inp)) )
