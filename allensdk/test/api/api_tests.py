
import unittest
from mock import MagicMock
from allensdk.api.api import Api
import allensdk.core.json_utilities as ju

class ApiTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ApiTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.api = Api()
    
    
    def tearDown(self):
        self.api = None
    
    
    def test_do_query_post(self):
        ju.read_url_post = \
            MagicMock(name='read_url_post',
                      return_value={ 'whatever': True })
               
        self.api.do_query(lambda *a, **k: 'http://localhost/%s' % (a[0]),
                          lambda d: d,
                          "wow",
                          post=True)
        
        ju.read_url_post.assert_called_once_with('http://localhost/wow')
    
    
    def test_do_query_get(self):
        ju.read_url_get = \
            MagicMock(name='read_url_get',
                      return_value={ 'whatever': True })
               
        self.api.do_query(lambda *a, **k: 'http://localhost/%s' % (a[0]),
                          lambda d: d,
                          "wow",
                          post=False)
        
        ju.read_url_get.assert_called_once_with('http://localhost/wow')
    
    
    def test_load_api_schema(self):
        ju.read_url_get = \
            MagicMock(name='read_url_get',
                      return_value={ 'whatever': True })
               
        self.api.load_api_schema()
        
        ju.read_url_get.assert_called_once_with('http://api.brain-map.org/api/v2/data/enumerate.json')
