# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2015-2018. Allen Institute. All rights reserved.
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
import numpy as np
import SimpleITK as sitk
import nrrd

from allensdk.core import sitk_utilities as su



@pytest.fixture(params=[1, 2, 3, 4])
def ncomponents(request):
    return request.param


@pytest.fixture(params=[ [10, 20], [10, 20, 30], [10, 20, 30], [10, 10, 10], [20, 20] ])
def size(request):
    return request.param


@pytest.fixture(params=[ lambda x: list(range(x+1))[1:], lambda x: [10] * x ])
def spacing(request):
    return request.param


@pytest.fixture(params=[ lambda x: list(range(x+1))[1:], lambda x: [5] * x ])
def origin(request):
    return request.param


@pytest.fixture(params=[ lambda x: np.eye(x).flatten() ])
def direction(request):
    return request.param


@pytest.fixture(scope='function')
def image(size, ncomponents, spacing, origin, direction):

    if ncomponents > 1:
        img = sitk.Image(size, sitk.sitkVectorUInt8, ncomponents)
    else:
        img = sitk.Image(size, sitk.sitkUInt8, ncomponents)

    ndim = len(size)

    spacing_val = spacing(ndim)
    img.SetSpacing(spacing_val)

    origin_val = origin(ndim)
    img.SetOrigin(origin_val)

    dir_val = direction(ndim)
    img.SetDirection(dir_val)

    return img, {'ncomponents': ncomponents, 
                 'size': size,
                 'spacing': spacing_val, 
                 'origin': origin_val, 
                 'direction': dir_val}


def test_get_sitk_image_information(image):

    obtained = su.get_sitk_image_information(image[0])
    for key, value in image[1].items():
        assert(np.allclose( obtained[key], value ))


def test_set_sitk_image_information_roundtrip(image):

    info = su.get_sitk_image_information(image[0])
    arr = sitk.GetArrayFromImage(image[0])

    new_image = sitk.GetImageFromArray(arr, info['ncomponents'] > 1)
    su.set_sitk_image_information(new_image, info)

    obtained = su.get_sitk_image_information(new_image)
    for key, value in info.items():
        assert(np.allclose( obtained[key], value ))


@pytest.mark.parametrize('act,dec,nc', [ ([10, 20], [20, 10], 1), 
                                         ([10, 20, 30], [30, 20, 10], 1),
                                         ([10, 20, 30, 3], [30, 20, 10], 3),
                                         ([10, 10, 10, 3], [10, 10, 10], 3 ) ])
def test_fix_array_dimensions(act, dec, nc):

    arr = np.zeros(act)
    obt = su.fix_array_dimensions(arr, nc)

    if nc == 1:
        assert(np.array_equal( obt.shape, dec ))
    else:
        assert(np.array_equal( obt.shape[:-1], dec ))

    assert( not np.isfortran(obt) )


def test_sitk_metaimage_roundtrip(tmpdir_factory, size):

    path = tmpdir_factory.mktemp('metaimage_io_test').join('dummy.mhd')
    
    array = np.random.rand(*size)

    su.write_ndarray_with_sitk(array, path)
    obt_image, obt_info = su.read_ndarray_with_sitk(path)

    assert(np.allclose( obt_image, array ))


def test_sitk_metaimage_vector_roundtrip(tmpdir_factory, size):

    path = tmpdir_factory.mktemp('metaimage_io_test').join('dummy.mhd')
    
    size = list(size) + [3]
    array = np.random.rand(*size)

    su.write_ndarray_with_sitk(array, path, ncomponents=3)
    obt_image, obt_info = su.read_ndarray_with_sitk(path)

    assert(np.allclose( obt_image, array ))
    assert( obt_info['ncomponents'] == 3 )


def test_sitk_nrrd_read(tmpdir_factory, size):
    
    path = tmpdir_factory.mktemp('nrrd_io_test').join('dummy.nrrd')
    array = np.random.rand(*size)
    
    nrrd.write(str(path), array)

    obt_image, obt_info = su.read_ndarray_with_sitk(path)

    assert(np.allclose( obt_image, array ))



def test_sitk_nrrd_write(tmpdir_factory, size):

    path = tmpdir_factory.mktemp('nrrd_io_test').join('dummy_again.nrrd')
    
    array = np.random.rand(*size)

    su.write_ndarray_with_sitk(array, path)
    obt_image, obt_info = nrrd.read(str(path))

    assert(np.allclose( obt_image, array ))
