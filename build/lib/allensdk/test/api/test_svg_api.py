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
#test AllenSDK svg api for download and show
from allensdk.api.queries.svg_api import SvgApi
import pytest
import json
from mock import MagicMock
import os

@pytest.fixture
def svg():
    sa = SvgApi()
    return sa
  
def test_build_query(svg):
    ####download true url
    download = True
    groups = None
    section_image_id = 21889
    returned_url = svg.build_query(section_image_id, groups,download)
    assert (returned_url == "http://api.brain-map.org/api/v2/svg_download/21889")
    
    ####download true with one group url
    download = True
    groups = [1]
    section_image_id = 21889
    returned_url = svg.build_query(section_image_id, groups,download)
    assert (returned_url == "http://api.brain-map.org/api/v2/svg_download/21889?groups=1")
    
    ####download true with groups url
    download = True
    groups = [1,2]
    section_image_id = 21889
    returned_url = svg.build_query(section_image_id, groups,download)
    assert (returned_url == "http://api.brain-map.org/api/v2/svg_download/21889?groups=1,2")
    
    ####download false url
    download = False
    groups = None
    section_image_id = 21889
    returned_url = svg.build_query(section_image_id, groups,download)
    assert (returned_url == "http://api.brain-map.org/api/v2/svg/21889")
    
    ####download false groups exist url
    download = False
    groups = [28]
    section_image_id = 21889
    returned_url = svg.build_query(section_image_id, groups,download)
    assert (returned_url == "http://api.brain-map.org/api/v2/svg/21889?groups=28")

def test_download_svg(svg):
    svg.retrieve_file_over_http = MagicMock(name='retrieve_file_over_http')
    section_image_id = 21889
    groups = None
    file_path = None
    
    svg.download_svg(section_image_id,groups,file_path)
    svg.retrieve_file_over_http.assert_called_with('http://api.brain-map.org/api/v2/svg_download/21889', '21889.svg')
    
                     
def test_get_svg(svg):
    svg.retrieve_xml_over_http = MagicMock(name='retrieve_xml_over_http')

    ####groups None
    section_image_id = 100960033
    groups = None

    svg.get_svg(section_image_id, groups)
    svg.retrieve_xml_over_http.assert_called_with("http://api.brain-map.org/api/v2/svg/100960033")
    
    ####groups in 28
    section_image_id = 100960033
    groups = [28]

    svg.get_svg(section_image_id, groups)
    svg.retrieve_xml_over_http.assert_called_with("http://api.brain-map.org/api/v2/svg/100960033?groups=28")
