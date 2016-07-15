import pytest
from mock import Mock, MagicMock
from allensdk.api.api import Api
import allensdk.core.json_utilities as ju
import urllib2
import socket


@pytest.fixture
def api():
    ju.read_url_post = \
        MagicMock(name='read_url_post',
                  return_value={ 'whatever': True })

    api = Api()
    
    return api


@pytest.fixture
def failed_download_api():
    error404 = urllib2.HTTPError(code=404,
                                 msg='not found',
                                 hdrs=Mock(),
                                 fp=Mock(),
                                 url='')
    
    urllib2.urlopen = Mock(side_effect=error404)
    
    api = Api()
    api._log.error = Mock()
    # api.retrieve_file_over_http = MagicMock()
    
    return api


@pytest.fixture
def timeout_download_api():
    error_timeout = socket.timeout
     
    urllib2.urlopen = Mock(side_effect=error_timeout)
     
    api = Api()
    api._log.error = Mock()
    # api.retrieve_file_over_http = MagicMock()
     
    return api


def test_timeout_download(timeout_download_api):
    with pytest.raises(socket.timeout) as e_info:
        timeout_download_api.retrieve_file_over_http('http://example.com/yo.zip',
                                                     '/tmp/testfile')
    
    # assert cp.assert_called_once_with('foobar')
    timeout_download_api._log.error.assert_called_once_with(
        "Timed out retrieving file from http://example.com/yo.zip")
    assert e_info.typename == 'timeout'


def test_failed_download(failed_download_api):
    with pytest.raises(urllib2.HTTPError) as e_info:
        failed_download_api.retrieve_file_over_http('http://example.com/yo.jpg',
                                                    '/tmp/testfile')
     
    # assert cp.assert_called_once_with('foobar')
    failed_download_api._log.error.assert_called_once_with(
        "Couldn't retrieve file from http://example.com/yo.jpg")
    assert e_info.typename == 'HTTPError'
    assert e_info.value.code == 404
 
 
def test_do_query_post(api):
    api.do_query(lambda *a, **k: 'http://localhost/%s' % (a[0]),
                 lambda d: d,
                 "wow",
                 post=True)
      
    ju.read_url_post.assert_called_once_with('http://localhost/wow')
  
  
def test_do_query_get(api):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={ 'whatever': True })
              
    api.do_query(lambda *a, **k: 'http://localhost/%s' % (a[0]),
                 lambda d: d,
                 "wow",
                 post=False)
       
    ju.read_url_get.assert_called_once_with('http://localhost/wow')
   
   
def test_load_api_schema(api):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={ 'whatever': True })
              
    api.load_api_schema()
       
    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/enumerate.json')
