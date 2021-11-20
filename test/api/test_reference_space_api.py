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
from mock import patch, MagicMock
import itertools as it
import numpy as np
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi as RSA


@pytest.fixture
def ref_space():
    rsa = RSA()
    
    return rsa


@pytest.fixture
def mock_nrrd():
    mocked_nrrd = MagicMock()
    mocked_nrrd.read = MagicMock(return_value=('mock_annotation_data',
                                               'mock_annotation_image'))
    return mocked_nrrd


def CCF_VERSIONS():
    return [RSA.CCF_2015,
            RSA.CCF_2016,
            RSA.CCF_2017]


def DATA_PATHS(): 
    return [RSA.AVERAGE_TEMPLATE,
            RSA.ARA_NISSL,
            RSA.MOUSE_2011,
            RSA.DEVMOUSE_2012,
            RSA.CCF_2015,
            RSA.CCF_2016, 
            RSA.CCF_2017]


def RESOLUTIONS():
    return [RSA.VOXEL_RESOLUTION_10_MICRONS,
            RSA.VOXEL_RESOLUTION_25_MICRONS,
            RSA.VOXEL_RESOLUTION_50_MICRONS,
            RSA.VOXEL_RESOLUTION_100_MICRONS]

MOCK_ANNOTATION_DATA = 'mock_annotation_data'
MOCK_ANNOTATION_IMAGE = 'mock_annotation_image'



def test_download_mouse_atlas_volume(ref_space):

    with patch.object(ref_space, 'retrieve_file_over_http') as mock_retrieve:
        with pytest.raises(RuntimeError):
            ref_space.download_mouse_atlas_volume('P56', 'Mouse_gridAnnotation', 'P56/gridAnnotation.mhd')    

    mock_retrieve.assert_called_once_with(
        'http://download.alleninstitute.org/informatics-archive/'\
        'current-release/mouse_annotation/'\
        'P56_Mouse_gridAnnotation.zip',
        'P56/gridAnnotation.mhd',
        zipped=True)


@pytest.mark.parametrize("data_path,resolution",
                         it.product(DATA_PATHS(),
                                    RESOLUTIONS()))
def test_download_volumetric_data(ref_space,
                                  data_path,
                                  resolution):
    cache_filename = "annotation_%d.nrrd" % (resolution)

    with patch.object(ref_space, "retrieve_file_over_http") as mock_retrieve:
        ref_space.download_volumetric_data(data_path,
                                           cache_filename,
                                           resolution)

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (data_path,
        resolution),
        cache_filename)


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_structure_mask(os_makedirs,
                                 nrrd_read,
                                 ref_space, 
                                 resolution):

    structure_id = 12

    with patch.object(ref_space, "retrieve_file_over_http") as mock_retrieve:
        a, b = ref_space.download_structure_mask(structure_id,
                                                 None, resolution,
                                                 '/path/to/foo.nrrd',
                                                 reader=nrrd_read)

    assert a
    assert b

    expected = 'http://download.alleninstitute.org/informatics-archive/'\
               'current-release/mouse_ccf/{0}/structure_masks/'\
               'structure_masks_{1}/structure_{2}.nrrd'.format(RSA.CCF_VERSION_DEFAULT, 
                                                               resolution, 
                                                               structure_id)
    mock_retrieve.assert_called_once_with(expected, '/path/to/foo.nrrd')
    os_makedirs.assert_any_call('/path/to')


@patch('allensdk.core.obj_utilities.read_obj', return_value=('mock_obj'))
@patch('os.makedirs')
def test_download_structure_mesh(os_makedirs,
                                 read_obj,
                                 ref_space):

    structure_id = 12

    with patch.object(ref_space, "retrieve_file_over_http") as mock_retrieve:
        a = ref_space.download_structure_mesh(structure_id,
                                              None, '/path/to/foo.obj',
                                              reader=read_obj)
    
    assert a == 'mock_obj'
    
    expected = 'http://download.alleninstitute.org/informatics-archive/'\
               'current-release/mouse_ccf/{0}/structure_meshes/'\
               '{1}.obj'.format(RSA.CCF_VERSION_DEFAULT, structure_id)

    mock_retrieve.assert_called_once_with(expected, '/path/to/foo.obj')
    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("ccf_version,resolution",
                         it.product(CCF_VERSIONS(),
                                    RESOLUTIONS()))
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_annotation_volume(os_makedirs,
                                    nrrd_read,
                                    ref_space,
                                    ccf_version,
                                    resolution):
    cache_file = '/path/to/annotation_%d.nrrd' % (resolution)

    with patch.object(ref_space, "retrieve_file_over_http") as mock_retrieve:
        ref_space.download_annotation_volume(
            ccf_version,
            resolution,
            cache_file,
            reader=nrrd_read)

    nrrd_read.assert_called_once_with(cache_file)

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (ccf_version, resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_annotation_volume_default(os_makedirs,
                                            nrrd_read,
                                            ref_space,
                                            resolution):
    with patch.object(ref_space, "retrieve_file_over_http") as mock_retrieve:
        a, b = ref_space.download_annotation_volume(
            None,
            resolution,
            '/path/to/annotation_%d.nrrd' % (resolution),
            reader=nrrd_read)

    assert a
    assert b

    print(mock_retrieve.call_args_list)

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/%s/annotation_%d.nrrd" % 
        (RSA.CCF_VERSION_DEFAULT, resolution),
        "/path/to/annotation_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')


@pytest.mark.parametrize("resolution",
                         RESOLUTIONS())
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch('os.makedirs')
def test_download_template_volume(os_makedirs,
                                  nrrd_read,
                                  ref_space,
                                  resolution):
    with patch.object(ref_space, "retrieve_file_over_http") as mock_retrieve:
        ref_space.download_template_volume(
            resolution,
            '/path/to/average_template_%d.nrrd' % (resolution),
            reader=nrrd_read)

    mock_retrieve.assert_called_once_with(
        "http://download.alleninstitute.org/informatics-archive/"
        "current-release/mouse_ccf/average_template/average_template_%d.nrrd" % 
        (resolution),
        "/path/to/average_template_%d.nrrd" % (resolution))

    os_makedirs.assert_any_call('/path/to')
