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
