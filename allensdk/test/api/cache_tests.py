import unittest
from mock import MagicMock
from allensdk.api.cache import Cache
from allensdk.api.queries.rma_api import RmaApi
import allensdk.core.json_utilities as ju
import pandas as pd
import pandas.io.json as pj


class CacheTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CacheTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.cache = Cache()
        self.api = RmaApi()
    
    
    def tearDown(self):
        self.cache = None
        self.api = None
    
    
    def test_wrap_json(self):
        msg = [ { 'whatever': True } ]
        
        ju.read_url_get = \
            MagicMock(name='read_url_get',
                      return_value={ 'msg': msg })
        ju.write = \
            MagicMock(name='write')
#        pj.read_json = \
#            MagicMock(name='read_json',
#                      return_value=msg)
        ju.read = \
            MagicMock(name='read',
                      return_value=pd.DataFrame(msg))
               
        df = self.cache.wrap(self.api.model_query,
                             'example.txt',
                             cache=True,
                             model='Hemisphere')
        
        self.assertTrue(df.loc[:,'whatever'][0])
        
        ju.read_url_get.assert_called_once_with('http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
        ju.write.assert_called_once_with('example.txt', msg)
        ju.read.assert_called_once_with('example.txt')
#        pj.read_json.assert_called_once_with('example.txt')
    
    
    def test_wrap_dataframe(self):
        msg = [ { 'whatever': True } ]
        
        ju.read_url_get = \
            MagicMock(name='read_url_get',
                      return_value={ 'msg': msg })
        ju.write = \
            MagicMock(name='write')
        pj.read_json = \
            MagicMock(name='read_json',
                      return_value=msg)
        
        json_data = self.cache.wrap(self.api.model_query,
                                    'example.txt',
                                    cache=True,
                                    return_dataframe=True,
                                    model='Hemisphere')
        
        self.assertTrue(json_data[0]['whatever'])
        
        ju.read_url_get.assert_called_once_with('http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
        ju.write.assert_called_once_with('example.txt', msg)
        pj.read_json.assert_called_once_with('example.txt', orient='records')