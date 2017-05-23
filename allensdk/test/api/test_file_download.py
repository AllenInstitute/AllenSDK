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
from mock import Mock, patch
from allensdk.api.cache import cacheable, Cache
from allensdk.config.manifest import Manifest
import allensdk.core.json_utilities as ju
import pandas.io.json as pj
import pandas as pd

try:
    import StringIO
except:
    import io as StringIO

try:
    reload
except NameError:
    try:
        from importlib import reload
    except ImportError:
        from imp import reload


@pytest.fixture(scope='module', autouse=True)
def mock_imports():
    with patch('nrrd.read',
               Mock(name='nrrd_read_file_mcm',
                    return_value=('mock_annotation_data',
                                  'mock_annotation_image'))) as nrrd_read:
        import allensdk.api.queries.mouse_connectivity_api
        reload(allensdk.api.queries.mouse_connectivity_api)
        from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi as MCA

    return nrrd_read, MCA


@pytest.fixture
def cache():
    return Cache()


_msg = [{'whatever': True}]
_pd_msg = pd.DataFrame(_msg)
_csv_msg = pd.read_csv(StringIO.StringIO(""",whatever
0,True
"""))

@pytest.fixture
def mca(mock_imports):
    _, MCA = mock_imports
    
    ju.read_url_get = \
        Mock(name='read_url_get',
             return_value={'msg': _msg})
    ju.write = \
        Mock(name='write')

    ju.read = \
        Mock(name='read',
             return_value=_pd_msg)

    pj.read_json = \
        Mock(name='read_json',
             return_value=_pd_msg)

    pd.DataFrame.to_csv = \
        Mock(name='to_csv')

    pd.DataFrame.read_csv = \
        Mock(name='read_csv',
             return_value=_csv_msg)

    Manifest.safe_mkdir = Mock(name='safe_mkdir')

    mca = MCA()
    mca.retrieve_file_over_http = Mock(name='retrieve_file_over_http')
    
    return mca


@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_download_lazy(mock_imports,
                            mca, 
                            cache,
                            file_exists):
    nrrd_read, MCA = mock_imports

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
        assert not mca.retrieve_file_over_http.called, 'server call not needed when file exists'
    else:
        mca.retrieve_file_over_http.assert_called_once_with(
            'http://download.alleninstitute.org/informatics-archive/annotation/ccf_2016/mouse_ccf/average_template/annotation_10.nrrd',
            'volumetric.nrrd')
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd_read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_download_server(mock_imports,
                              mca,
                              cache,
                             file_exists):
    nrrd_read, MCA = mock_imports

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

    mca.retrieve_file_over_http.assert_called_once_with(
        'http://download.alleninstitute.org/informatics-archive/annotation/ccf_2016/mouse_ccf/average_template/annotation_10.nrrd',
        'volumetric.nrrd')
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd_read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_download_cached_file(mock_imports,
                                   mca,
                                   cache,
                                   file_exists):
    nrrd_read, MCA = mock_imports

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

    assert not mca.retrieve_file_over_http.called, 'server should not have been called'
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd_read.assert_called_once_with('volumetric.nrrd')


@pytest.mark.parametrize("file_exists",
                         (True, False))
def test_file_kwarg(mock_imports,
                    mca,
                    cache,
                    file_exists):
    nrrd_read, MCA = mock_imports

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

    assert not mca.retrieve_file_over_http.called, 'server should not have been called'
    assert not Manifest.safe_mkdir.called, 'safe_mkdir should not have been called.'
    nrrd_read.assert_called_once_with('file.nrrd')


@pytest.mark.run('last')
def test_cleanup():
    import allensdk.api.queries.mouse_connectivity_api
    reload(allensdk.api.queries.mouse_connectivity_api)
    import allensdk.core.mouse_connectivity_cache
    reload(allensdk.core.mouse_connectivity_cache)
