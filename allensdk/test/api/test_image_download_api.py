import pytest
from mock import MagicMock
from allensdk.api.queries.image_download_api import ImageDownloadApi


@pytest.fixture
def image_api():
    image_api = ImageDownloadApi()
    
    image_api.retrieve_file_over_http = \
        MagicMock(name='retrieve_file_over_http')
    
    return image_api
    
    
def test_api_doc_url_download_image_downsampled(image_api):
    '''
    Notes
    -----
    See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
    , link labeled 'Download image downsampled by factor of 6 using default thresholds'.
    '''
    path = '126862575.jpg'
    
    section_image_id = 126862575
    image_api.download_section_image(section_image_id,
                                     downsample=6,
                                     range=[0,932, 0,1279, 0,4095])
    
    image_api.retrieve_file_over_http.assert_called_once_with(
        "http://api.brain-map.org/api/v2/section_image_download/126862575"
        "?downsample=6&range=0,932,0,1279,0,4095",
        path)


def test_api_doc_url_download_image_full_resolution(image_api):
    '''
    Notes
    -----
    See: `Experimental Overview and Metadata `<http://help.brain-map.org/display/mouseconnectivity/API#API-ExperimentalOverviewandMetadata>_
    , link labeled 'Download a region of interest at full resolution using default thresholds'.
    '''
    expected = 'http://api.brain-map.org/api/v2/section_image_download/126862575?left=19045&top=11684&width=1000&height=1000&range=0,932,0,1279,0,4095'
    path = '126862575.jpg'

    image_api.retrieve_file_over_http = \
        MagicMock(name='retrieve_file_over_http')
    
    section_image_id = 126862575
    image_api.download_section_image(section_image_id,
                                     left=19045,
                                     top=11684,
                                     width=1000,
                                     height=1000,
                                     range=[0,932, 0,1279, 0,4095])
    
    image_api.retrieve_file_over_http.assert_called_once_with(expected, path)
