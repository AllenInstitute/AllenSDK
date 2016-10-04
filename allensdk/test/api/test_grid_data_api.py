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
from mock import MagicMock
from allensdk.api.queries.grid_data_api import GridDataApi


@pytest.fixture
def grid_data():
    gda = GridDataApi()
    gda.retrieve_file_over_http = \
        MagicMock(name='retrieve_file_over_http')

    return gda


def test_api_doc_url_download_expression_grid(grid_data):
    '''Url to download the 200um density volume
       for the Mouse Brain Atlas SectionDataSet 69816930.

    Notes
    -----
    See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
    , example 'Download the 200um density volume for the Mouse Brain Atlas SectionDataSet 69816930'.
    '''
    path = '69816930.zip'
    section_data_set_id = 69816930
    grid_data.download_expression_grid_data(section_data_set_id)
    expected = 'http://api.brain-map.org/grid_data/download/69816930'
    grid_data.retrieve_file_over_http.assert_called_once_with(expected, path)


def test_api_doc_url_download_expression_grid_energy_intensity(grid_data):
    '''Url to download the 200um energy and intensity volumes for Mouse Brain Atlas SectionDataSet 69816930.

    Notes
    -----
    See `Downloading 3-D Expression Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DEXPRESSIONGRIDDATA>`_
    , example 'Download the 200um energy and intensity volumes for Mouse Brain Atlas SectionDataSet 69816930'.

    The id in the example url doesn't match the caption.
    '''
    path = '183282970.zip'
    section_data_set_id = 183282970
    include = ['energy', 'intensity']
    grid_data.download_expression_grid_data(section_data_set_id,
                                            include=include)

    grid_data.retrieve_file_over_http.assert_called_once_with(
        "http://api.brain-map.org/grid_data/download/183282970"
        "?include=energy,intensity",
        path)


def test_api_doc_url_projection_grid(grid_data):
    '''Url to download the 100um density volume for the Mouse Connectivity Atlas SectionDataSet 181777177.

    Notes
    -----
    See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA>`_
    , example 'Download the 100um density volume for the Mouse Connectivity Atlas SectionDataSet 181777177'.
    '''
    path = '181777177.nrrd'
    section_data_set_id = 181777177
    grid_data.download_projection_grid_data(section_data_set_id)
    expected = 'http://api.brain-map.org/grid_data/download_file/181777177'
    grid_data.retrieve_file_over_http.assert_called_once_with(expected, path)


def test_api_doc_url_projection_grid_injection_fraction_resolution(grid_data):
    '''Url to download the 25um injection_fraction volume for Mouse Connectivity Atlas SectionDataSet 181777177.

    Notes
    -----
    See `Downloading 3-D Projection Grid Data <http://help.brain-map.org/display/api/Downloading+3-D+Expression+Grid+Data#Downloading3-DExpressionGridData-DOWNLOADING3DPROJECTIONGRIDDATA>`_
    , example 'Download the 25um injection_fraction volume for Mouse Connectivity Atlas SectionDataSet 181777177'.
    '''
    section_data_set_id = 181777177
    path = 'id.nrrd'
    grid_data.download_projection_grid_data(section_data_set_id,
                                            [grid_data.INJECTION_FRACTION],
                                            resolution=25,
                                            save_file_path=path)

    grid_data.retrieve_file_over_http.assert_called_once_with(
        "http://api.brain-map.org/grid_data/download_file/181777177"
        "?image=injection_fraction&resolution=25",
        path)
