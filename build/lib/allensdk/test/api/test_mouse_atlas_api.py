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

from mock import MagicMock, patch
import pytest

from allensdk.api.queries.mouse_atlas_api import MouseAtlasApi as MAA



@pytest.fixture
def atlas():
    maa = MAA()
    return maa


@patch.object(MAA, "json_msg_query")
def test_get_genes(mock_query, atlas):

    expected = 'http://api.brain-map.org/api/v2/data/query.json?'\
               'q=model::Gene,rma::criteria,[organism_id$in2],rma::include,chromosome,'\
               'rma::options[num_rows$eq2000][start_row$eq0][order$eq\'id\'][count$eqfalse]'

    for result in atlas.get_genes():
        pass
    
    mock_query.assert_called_once_with(expected)


@patch.object(MAA, "json_msg_query")
def test_get_section_data_sets(mock_query, atlas):

    expected = 'http://api.brain-map.org/api/v2/data/query.json?'\
               'q=model::SectionDataSet,rma::criteria,products[id$in1],rma::include,genes,'\
               'rma::options[num_rows$eq2000][start_row$eq0][order$eq\'id\'][count$eqfalse]'

    for result in atlas.get_section_data_sets():
        pass
    
    mock_query.assert_called_once_with(expected)


def test_download_expression_density(atlas):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        with pytest.raises(RuntimeError):
            atlas.download_expression_density('file.name', 12345)

            gda.assert_called_once_with(
                'http://api.brain-map.org/grid_data/download/'\
                '12345?include=density')
                

def test_download_expression_intensity(atlas):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        with pytest.raises(RuntimeError):
            atlas.download_expression_intensity('file.name', 12345)

            gda.assert_called_once_with(
                'http://api.brain-map.org/grid_data/download/'\
                '12345?include=intensity')


def test_download_expression_energy(atlas):
    with patch('allensdk.api.api.Api.retrieve_file_over_http') as gda:
        with pytest.raises(RuntimeError):
            atlas.download_expression_energy('file.name', 12345)

            gda.assert_called_once_with(
                'http://api.brain-map.org/grid_data/download/'\
                '12345?include=energy')
