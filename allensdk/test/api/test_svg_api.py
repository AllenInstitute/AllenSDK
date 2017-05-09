#test AllenSDK svg api for download and show
from allensdk.api.queries.svg_api import SvgApi
import pytest
import json
from mock import MagicMock
import StringIO
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
