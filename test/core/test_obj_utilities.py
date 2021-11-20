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

import numpy as np

import pytest
from mock import MagicMock, mock_open, patch

from allensdk.core.obj_utilities import read_obj, parse_obj


@pytest.fixture
def wavefront_obj():
    return '''

v 8578 5484.96 5227.57
v 8509.2 5487.54 5237.07
v 8564.38 5522.13 5220.41
v 8631.93 5497.82 5228.33
v 8517.88 5542.95 5234.53
v 8615.26 5563.22 5224.48

# i'm a comment!

vn -0.0247061 -0.352726 -0.935401
vn -0.235489 -0.190095 -0.953105
vn -0.0880336 -0.0323767 -0.995591
vn 0.122706 -0.209891 -0.969994
vn -0.343738 0.217978 -0.913416
vn 0.0753706 0.16324 -0.983703

I should be a comment, but am not

f 1//1 2//2 3//3 
f 4//4 1//1 3//3 
f 3//3 2//2 5//5 
f 6//6 3//3 5//5 

    '''


def test_read_obj(wavefront_obj):

    path = 'path!'

    # need to patch the version in allensdk.api.cache because of import x from y syntax above
    with patch( 'allensdk.core.obj_utilities.open', mock_open(read_data=wavefront_obj), create=True ) as p:
        obt = read_obj(path)
        p.assert_called_with(path, 'r')
        assert( obt is not None )        


def test_parse_obj(wavefront_obj):

    lines = wavefront_obj.split('\n')
    vertices, vertex_normals, face_vertices, face_normals = parse_obj(lines)
    
    assert(np.allclose( face_vertices, face_normals ))
    assert(np.allclose( face_vertices[2, :], [2, 1, 4] ))
    assert(np.allclose( vertices[1, :], [8509.2, 5487.54, 5237.07] ))
    assert(np.allclose( vertex_normals[2, :], [-0.0880336, -0.0323767, -0.995591] ))
