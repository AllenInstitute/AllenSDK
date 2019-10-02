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
import pytest
from mock import Mock, patch
from allensdk.api.cache import cacheable, Cache
from allensdk.config.manifest import Manifest
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi as MCA


try:
    import StringIO
except:
    import io as StringIO


@pytest.fixture
def mca():
    return MCA()


@pytest.fixture
def cache():
    return Cache()


@pytest.mark.parametrize("file_exists", (True, False))
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch.object(Manifest, 'safe_mkdir')
def test_file_download_lazy(nrrd_read, safe_mkdir, mca, cache, file_exists):
    with patch.object(mca, "retrieve_file_over_http") as mock_retrieve:
        @cacheable(strategy='lazy',
                reader=nrrd_read,
                pathfinder=Cache.pathfinder(file_name_position=3,
                                            secondary_file_name_position=1))
        def download_volumetric_data(data_path,
                                    file_name,
                                    voxel_resolution=None,
                                    save_file_path=None,
                                    release=None,
                                    coordinate_framework=None):
            url = mca.build_volumetric_data_download_url(data_path,
                                                        file_name,
                                                        voxel_resolution,
                                                        release,
                                                        coordinate_framework)

            mca.retrieve_file_over_http(url, save_file_path)

        with patch('os.path.exists',
                Mock(name="os.path.exists",
                        return_value=file_exists)) as mkdir:
            nrrd_read.reset_mock()
            download_volumetric_data(MCA.AVERAGE_TEMPLATE,
                                    'annotation_10.nrrd',
                                    MCA.VOXEL_RESOLUTION_10_MICRONS,
                                    'volumetric.nrrd',
                                    MCA.CCF_2016,
                                    strategy='lazy')

        if file_exists:
            assert not mock_retrieve.called, 'server call not needed when file exists'
        else:
            mock_retrieve.assert_called_once_with(
                'http://download.alleninstitute.org/informatics-archive/annotation/ccf_2016/mouse_ccf/average_template/annotation_10.nrrd',
                'volumetric.nrrd')
        assert not safe_mkdir.called, 'safe_mkdir should not have been called.'
        nrrd_read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists", (True, False))
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch.object(Manifest, 'safe_mkdir')
def test_file_download_server(nrrd_read, safe_mkdir, mca, cache, file_exists):
    with patch.object(mca, "retrieve_file_over_http") as mock_retrieve:
        @cacheable(reader=nrrd_read,
                pathfinder=Cache.pathfinder(file_name_position=3,
                                            secondary_file_name_position=1))
        def download_volumetric_data(data_path,
                                    file_name,
                                    voxel_resolution=None,
                                    save_file_path=None,
                                    release=None,
                                    coordinate_framework=None):
            url = mca.build_volumetric_data_download_url(data_path,
                                                        file_name,
                                                        voxel_resolution,
                                                        release,
                                                        coordinate_framework)

            mca.retrieve_file_over_http(url, save_file_path)

        with patch('os.path.exists',
                Mock(name="os.path.exists",
                        return_value=file_exists)) as mkdir:
            nrrd_read.reset_mock()
            
            download_volumetric_data(MCA.AVERAGE_TEMPLATE,
                                    'annotation_10.nrrd',
                                    MCA.VOXEL_RESOLUTION_10_MICRONS,
                                    'volumetric.nrrd',
                                    MCA.CCF_2016,
                                    strategy='create')

        mock_retrieve.assert_called_once_with(
            'http://download.alleninstitute.org/informatics-archive/annotation/ccf_2016/mouse_ccf/average_template/annotation_10.nrrd',
            'volumetric.nrrd')
        assert not safe_mkdir.called, 'safe_mkdir should not have been called.'
        nrrd_read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists", (True, False))
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch.object(Manifest, 'safe_mkdir')
def test_file_download_cached_file(nrrd_read, safe_mkdir, mca, cache, file_exists):
    with patch.object(mca, "retrieve_file_over_http") as mock_retrieve:
        @cacheable(reader=nrrd_read,
                pathfinder=Cache.pathfinder(file_name_position=3,
                                            secondary_file_name_position=1))
        def download_volumetric_data(data_path,
                                    file_name,
                                    voxel_resolution=None,
                                    save_file_path=None,
                                    release=None,
                                    coordinate_framework=None):
            url = mca.build_volumetric_data_download_url(data_path,
                                                        file_name,
                                                        voxel_resolution,
                                                        release,
                                                        coordinate_framework)

            mca.retrieve_file_over_http(url, save_file_path)

        with patch('os.path.exists',
                Mock(name="os.path.exists",
                        return_value=file_exists)) as mkdir:
            nrrd_read.reset_mock()

            download_volumetric_data(MCA.AVERAGE_TEMPLATE,
                                    'annotation_10.nrrd',
                                    MCA.VOXEL_RESOLUTION_10_MICRONS,
                                    'volumetric.nrrd',
                                    MCA.CCF_2016,
                                    strategy='file')

        assert not mock_retrieve.called, 'server should not have been called'
        assert not safe_mkdir.called, 'safe_mkdir should not have been called.'
        nrrd_read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists", (True, False))
@patch("nrrd.read", return_value=('mock_annotation_data',
                                  'mock_annotation_image'))
@patch.object(Manifest, 'safe_mkdir')
def test_file_kwarg(nrrd_read, safe_mkdir, mca, cache, file_exists):
    with patch.object(mca, "retrieve_file_over_http") as mock_retrieve:
        @cacheable(reader=nrrd_read,
                pathfinder=Cache.pathfinder(file_name_position=3,
                                            secondary_file_name_position=1,
                                            path_keyword='save_file_path'))
        def download_volumetric_data(data_path,
                                    file_name,
                                    voxel_resolution=None,
                                    save_file_path=None,
                                    release=None,
                                    coordinate_framework=None):
            url = mca.build_volumetric_data_download_url(data_path,
                                                        file_name,
                                                        voxel_resolution,
                                                        release,
                                                        coordinate_framework)

            mca.retrieve_file_over_http(url, save_file_path)

        with patch('os.path.exists',
                Mock(name="os.path.exists",
                        return_value=file_exists)) as mkdir:
            nrrd_read.reset_mock()

            download_volumetric_data(MCA.AVERAGE_TEMPLATE,
                                    'annotation_10.nrrd',
                                    MCA.VOXEL_RESOLUTION_10_MICRONS,
                                    'volumetric.nrrd',
                                    MCA.CCF_2016,
                                    strategy='file',
                                    save_file_path='file.nrrd' )

        assert not mock_retrieve.called, 'server should not have been called'
        assert not safe_mkdir.called, 'safe_mkdir should not have been called.'
        nrrd_read.assert_called_once_with('file.nrrd')
