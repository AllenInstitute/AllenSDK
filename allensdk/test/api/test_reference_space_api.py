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
from mock import patch, Mock
import itertools as it
import numpy as np


try:
    reload
except NameError:
    try:
        from importlib import reload
    except ImportError:
        from imp import reload

        
@pytest.fixture
def nrrd_read():
    with patch('nrrd.read',
               Mock(name='nrrd_read_file_mcm',
                    return_value=('mock_annotation_data',
                                  'mock_annotation_image'))) as nrrd_read:
        import allensdk.api.queries.reference_space_api as RSA
        reload(RSA)

    return nrrd_read


@pytest.fixture
def read_obj():
    with patch('allensdk.core.obj_utilities.read_obj', 
               Mock(name='read_obj_file_mcm', return_value=('mock_obj'))) as read_obj:
        import allensdk.api.queries.reference_space_api as RSA
        reload(RSA)

    return read_obj


@pytest.fixture
def RSA(nrrd_read):
    import allensdk.api.queries.reference_space_api as RSA

    download_link = '/path/to/link'
    RSA.ReferenceSpaceApi.do_query = Mock(return_value=download_link)
    RSA.ReferenceSpaceApi.json_msg_query = Mock(name='json_msg_query')
    RSA.ReferenceSpaceApi.retrieve_file_over_http = \
        Mock(name='retrieve_file_over_http')
    
    return RSA

@pytest.fixture
def ref_space(RSA):
    rsa = RSA.ReferenceSpaceApi()
    
    return rsa


def CCF_VERSIONS():
    from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
     
    return [ReferenceSpaceApi.CCF_2015,
            ReferenceSpaceApi.CCF_2016,
            ReferenceSpaceApi.CCF_2017]


def DATA_PATHS(): 
    from allensdk.api.queries.reference_space_api import ReferenceSpaceApi

    return [ReferenceSpaceApi.AVERAGE_TEMPLATE,
            ReferenceSpaceApi.ARA_NISSL,
            ReferenceSpaceApi.MOUSE_2011,
            ReferenceSpaceApi.DEVMOUSE_2012,
            ReferenceSpaceApi.CCF_2015,
            ReferenceSpaceApi.CCF_2016, 
            ReferenceSpaceApi.CCF_2017]


def RESOLUTIONS():
    from allensdk.api.queries.reference_space_api import ReferenceSpaceApi

    return [ReferenceSpaceApi.VOXEL_RESOLUTION_10_MICRONS,
            ReferenceSpaceApi.VOXEL_RESOLUTION_25_MICRONS,
            ReferenceSpaceApi.VOXEL_RESOLUTION_50_MICRONS,
            ReferenceSpaceApi.VOXEL_RESOLUTION_100_MICRONS]

MOCK_ANNOTATION_DATA = 'mock_annotation_data'
MOCK_ANNOTATION_IMAGE = 'mock_annotation_image'


@pytest.mark.parametrize("data_path,resolution",
                         it.product(DATA_PATHS(),
                                    RESOLUTIONS()))
def test_download_volumetric_data(nrrd_read,
                                  ref_space,
                                  data_path,
                                  resolution):
    cache_filename = "annotation_%d.nrrd" % (resolution)

    nrrd_read.reset_mock()
    ref_space.retrieve_file_over_http.reset_mock()

    ref_space.download_volumetric_data(data_path,
                                          cache_filename,
                                          resolution)

    ref_space.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (data_path,
         resolution),
        cache_filename)


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch('os.makedirs')
def test_download_structure_mask(os_makedirs, 
                                 nrrd_read, 
                                 RSA, 
                                 ref_space, 
                                 resolution):

    structure_id = 12

    ref_space.retrieve_file_over_http.reset_mock()

    a, b = ref_space.download_structure_mask(structure_id, None, resolution, '/path/to/foo.nrrd')
    
    assert a
    assert b
    
    expected = 'http://download.alleninstitute.org/informatics-archive/'\
               'current-release/mouse_ccf/{0}/structure_masks/'\
               'structure_masks_{1}/structure_{2}.nrrd'.format(RSA.ReferenceSpaceApi.CCF_VERSION_DEFAULT, 
                                                               resolution, 
                                                               structure_id)
    ref_space.retrieve_file_over_http.assert_called_once_with(expected, '/path/to/foo.nrrd')
    os.makedirs.assert_any_call('/path/to')


@patch('os.makedirs')
def test_download_structure_mesh(os_makedirs, read_obj, RSA, ref_space):

    structure_id = 12

    ref_space.retrieve_file_over_http.reset_mock()

    a = ref_space.download_structure_mesh(structure_id, None, '/path/to/foo.obj', reader=read_obj)
    
    assert a == 'mock_obj'
    
    expected = 'http://download.alleninstitute.org/informatics-archive/'\
               'current-release/mouse_ccf/{0}/structure_meshes/'\
               '{1}.obj'.format(RSA.ReferenceSpaceApi.CCF_VERSION_DEFAULT, structure_id)

    ref_space.retrieve_file_over_http.assert_called_once_with(expected, '/path/to/foo.obj')
    os.makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("ccf_version,resolution",
                         it.product(CCF_VERSIONS(),
                                    RESOLUTIONS()))
@patch('os.makedirs')
def test_download_annotation_volume(os_makedirs,
                                    nrrd_read,
                                    ref_space,
                                    ccf_version,
                                    resolution):
    nrrd_read.reset_mock()
    ref_space.retrieve_file_over_http.reset_mock()

    cache_file = '/path/to/annotation_%d.nrrd' % (resolution)

    ref_space.download_annotation_volume(
        ccf_version,
        resolution,
        cache_file)

    nrrd_read.assert_called_once_with(cache_file)

    ref_space.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (ccf_version,
         resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch('os.makedirs')
def test_download_annotation_volume_default(os_makedirs,
                                            nrrd_read,
                                            RSA,
                                            ref_space,
                                            resolution):
    ref_space.retrieve_file_over_http.reset_mock()

    a, b = ref_space.download_annotation_volume(
        None,
        resolution,
        '/path/to/annotation_%d.nrrd' % (resolution),
        reader=nrrd_read)
    
    assert a
    assert b

    print(ref_space.retrieve_file_over_http.call_args_list)

    ref_space.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (RSA.ReferenceSpaceApi.CCF_VERSION_DEFAULT,
         resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch('os.makedirs')
def test_download_template_volume(os_makedirs,
                                  ref_space,
                                  resolution):
    ref_space.retrieve_file_over_http.reset_mock()

    ref_space.download_template_volume(
        resolution,
        '/path/to/average_template_%d.nrrd' % (resolution))

    ref_space.retrieve_file_over_http.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/average_template/average_template_%d.nrrd" % 
        (resolution),
        "/path/to/average_template_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')
