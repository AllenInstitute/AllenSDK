import pytest
from mock import MagicMock
from allensdk.api.cache import Cache
from allensdk.api.queries.rma_api import RmaApi
import allensdk.core.json_utilities as ju
import pandas as pd
import pandas.io.json as pj


@pytest.fixture
def cache():
    return Cache()

@pytest.fixture
def rma():
    return RmaApi()

def test_wrap_json(rma, cache):
    msg = [ { 'whatever': True } ]
    
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={ 'msg': msg })
    ju.write = \
        MagicMock(name='write')
    
    ju.read = \
        MagicMock(name='read',
                  return_value=pd.DataFrame(msg))
           
    df = cache.wrap(rma.model_query,
                    'example.txt',
                    cache=True,
                    model='Hemisphere')
    
    assert df.loc[:,'whatever'][0]
    
    ju.read_url_get.assert_called_once_with('http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    ju.write.assert_called_once_with('example.txt', msg)
    ju.read.assert_called_once_with('example.txt')


def test_wrap_dataframe(rma, cache):
    msg = [ { 'whatever': True } ]
     
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={ 'msg': msg })
    ju.write = \
        MagicMock(name='write')
    pj.read_json = \
        MagicMock(name='read_json',
                  return_value=msg)
     
    json_data = cache.wrap(rma.model_query,
                           'example.txt',
                           cache=True,
                           return_dataframe=True,
                           model='Hemisphere')
     
    assert json_data[0]['whatever']
     
    ju.read_url_get.assert_called_once_with('http://api.brain-map.org/api/v2/data/query.json?q=model::Hemisphere')
    ju.write.assert_called_once_with('example.txt', msg)
    pj.read_json.assert_called_once_with('example.txt', orient='records')