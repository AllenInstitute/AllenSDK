import pytest
from mock import MagicMock
from allensdk.api.api import Api
import allensdk.core.json_utilities as ju


@pytest.fixture
def api():
    ju.read_url_post = \
        MagicMock(name='read_url_post',
                  return_value={ 'whatever': True })

    api = Api()
    
    return api
    

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
