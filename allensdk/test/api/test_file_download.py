# Copyright 2016 Allen Institute for Brain Science
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


import pytest
from mock import MagicMock, patch
from allensdk.api.cache import Cache
from allensdk.config.manifest import file_download, Manifest
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd
import StringIO
import nrrd


@pytest.fixture
def cache():
    return Cache()


_msg = [{'whatever': True}]
_pd_msg = pd.DataFrame(_msg)
_csv_msg = pd.read_csv(StringIO.StringIO(""",whatever
0,True
"""))

@pytest.fixture
def mca():
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': _msg})
    ju.write = \
        MagicMock(name='write')

    ju.read = \
        MagicMock(name='read',
                  return_value=_pd_msg)

    pj.read_json = \
        MagicMock(name='read_json',
                  return_value=_pd_msg)

    pd.DataFrame.to_csv = \
        MagicMock(name='to_csv')

    pd.DataFrame.read_csv = \
        MagicMock(name='read_csv',
                  return_value=_csv_msg)

    Manifest.safe_mkdir = MagicMock(name='safe_mkdir')
    
    nrrd.read = MagicMock(name='nrrd.read',
                          return_value=('nrrd_file_a',
                                        'nrrd_file_b'))

    mca = MouseConnectivityApi()
    mca.retrieve_file_over_http = MagicMock(name='retrieve_file_over_http')
    
    return mca

@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_download_lazy(mca, cache,
                            file_exists):
    @file_download(reader='nrrd',
                   file_name_position=3,
                   secondary_file_name_position=1)
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

    with patch('os.path.exists', MagicMock(name="os.path.exists",
                                           return_value=file_exists)) as mkdir:
        download_volumetric_data(MouseConnectivityApi.AVERAGE_TEMPLATE,
                                 'annotation_10.nrrd',
                                 MouseConnectivityApi.VOXEL_RESOLUTION_10_MICRONS,
                                 'volumetric.nrrd',
                                 MouseConnectivityApi.CCF_2016,
                                 query_strategy='lazy')

    if file_exists:
        assert not mca.retrieve_file_over_http.called, 'server call not needed when file exists'
    else:
        mca.retrieve_file_over_http.assert_called_once_with(
            'http://download.alleninstitute.org/informatics-archive/annotation/ccf_2016/mouse_ccf/average_template/annotation_10.nrrd',
            'volumetric.nrrd')
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd.read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_download_server(mca, cache,
                             file_exists):
    @file_download(reader='nrrd',
                   file_name_position=3,
                   secondary_file_name_position=1)
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

    with patch('os.path.exists', MagicMock(name="os.path.exists",
                                           return_value=file_exists)) as mkdir:
        download_volumetric_data(MouseConnectivityApi.AVERAGE_TEMPLATE,
                                 'annotation_10.nrrd',
                                 MouseConnectivityApi.VOXEL_RESOLUTION_10_MICRONS,
                                 'volumetric.nrrd',
                                 MouseConnectivityApi.CCF_2016,
                                 query_strategy='server')

    mca.retrieve_file_over_http.assert_called_once_with(
        'http://download.alleninstitute.org/informatics-archive/annotation/ccf_2016/mouse_ccf/average_template/annotation_10.nrrd',
        'volumetric.nrrd')
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd.read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_download_cached_file(mca, cache,
                                   file_exists):
    @file_download(reader='nrrd',
                   file_name_position=3,
                   secondary_file_name_position=1)
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

    with patch('os.path.exists', MagicMock(name="os.path.exists",
                                           return_value=file_exists)) as mkdir:
        download_volumetric_data(MouseConnectivityApi.AVERAGE_TEMPLATE,
                                 'annotation_10.nrrd',
                                 MouseConnectivityApi.VOXEL_RESOLUTION_10_MICRONS,
                                 'volumetric.nrrd',
                                 MouseConnectivityApi.CCF_2016,
                                 query_strategy='file')

    assert not mca.retrieve_file_over_http.called, 'server should not have been called'
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd.read.assert_called_once_with('volumetric.nrrd')
