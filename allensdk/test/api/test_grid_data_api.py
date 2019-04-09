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
from mock import MagicMock, patch
from allensdk.api.queries.grid_data_api import GridDataApi


@pytest.fixture
def grid_data():
    gda = GridDataApi()
    gda.retrieve_file_over_http = \
        MagicMock(name='retrieve_file_over_http')

    return gda


def test_download_gene_expression_grid_data(grid_data):

    path = '69816930/density.mhd'
    section_data_set_id = 69816930
    volume_type = 'density'

    grid_data.download_gene_expression_grid_data(section_data_set_id, volume_type, path)
    expected = 'http://api.brain-map.org/grid_data/download/69816930?include=density'
    grid_data.retrieve_file_over_http.assert_called_once_with(expected, path, zipped=True)


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


def test_download_deformation_field(grid_data):
    grid_data.model_query = MagicMock(
        name='model_query', 
        return_value=[
            {'well_known_file_type': {'name': 'DeformationFieldHeader'}, 'id': 123}, 
            {'well_known_file_type': {'name': 'DeformationFieldVoxels'}, 'id': 456}
        ]
    )

    grid_data.download_deformation_field(789)

    grid_data.retrieve_file_over_http.assert_any_call('http://api.brain-map.org/api/v2/well_known_file_download/123', '789_dfmfld.mhd')
    grid_data.retrieve_file_over_http.assert_any_call('http://api.brain-map.org/api/v2/well_known_file_download/456', '789_dfmfld.raw')


def test_download_alignment3d(grid_data):
    grid_data.json_msg_query = MagicMock(
        name='json_msg_query',
        return_value=[{'alignment3d': 'foo'}]
    )

    obtained = grid_data.download_alignment3d(123)
    assert 'foo' == obtained
    grid_data.json_msg_query.assert_called_once_with((
        'http://api.brain-map.org/api/v2/data/query.json?q='
        'model::SectionDataSet[id$eq123],'
        'rma::include,alignment3d,'
        'rma::options[num_rows$eq\'all\'][count$eqfalse]'
    ))
