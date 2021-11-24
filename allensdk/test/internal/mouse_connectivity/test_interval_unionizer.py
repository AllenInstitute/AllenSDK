from __future__ import division

import numpy as np
import pytest
import mock
from six import iteritems

from allensdk.internal.mouse_connectivity.interval_unionize.interval_unionizer \
    import IntervalUnionizer


@pytest.fixture(scope='function')
def annotation():

    annot = np.zeros((10, 10, 10))
    annot[:, :, 4:] = 1  # 4 off, 6 on, ...
    annot[5:8, :, :] = 2  # solid block from 500:800
    
    return annot 
    
    
def test_init():
    
    iu = IntervalUnionizer([1, 2, 3])
    assert( np.allclose(iu.exclude_structure_ids, [1, 2, 3]) )
    
    iu = IntervalUnionizer()
    assert( sum(iu.exclude_structure_ids) == 0 )
    

def test_setup_interval_map(annotation):

    bounds_exp = {1: (280, 700), 2: (700, 1000)}
    
    iu = IntervalUnionizer()
    iu.setup_interval_map(annotation)
    
    for k, v in iteritems(iu.interval_map):
        assert( np.allclose(v, bounds_exp[k]) )
        
    
def test_extract_data():
    
    iu = IntervalUnionizer()
    
    with pytest.raises(NotImplementedError):
        iu.extract_data('olive', 'reticulated', 'western_woma')
        
        
def test_propagate_record():
    
    with pytest.raises(NotImplementedError):
        IntervalUnionizer.propagate_record('olive', 'reticulated')
        
        
def test_propagate_unionizes():

    uns = {1: {'a': 1, 'b': 2}, 
           2: {'a': 2, 'b': 3}, 
           3: {'a': 3, 'b': 4}}
           
    amap = {1: [1, 2], 2: [2], 3: [3]}

    def dummy_prop(cls, c, a):
        return {k: c[k] + a[k] for k in c}

    IntervalUnionizer.propagate_record = classmethod(dummy_prop)
    ou = IntervalUnionizer.propagate_unionizes(uns, amap)

    assert( ou[3]['a'] == 3 )
    assert( ou[2]['b'] == 5 )
    assert( ou[1]['a'] == 1 )
    
    
def test_postprocess_unionizes():
    
    iu = IntervalUnionizer()
    with pytest.raises(NotImplementedError):
        iu.postprocess_unionizes('foo')
        
        
def test_sort_data_arrays():

    data_arrays = {1: np.arange(10), 2: np.arange(10, 20)}
    sort = np.array([3, 1, 5, 2, 6, 4, 7, 8, 9, 0])
    
    iu = IntervalUnionizer()
    iu.sort = sort
    
    obt = iu.sort_data_arrays(data_arrays)
    
    assert( np.allclose(obt[1], sort) )
    assert( np.allclose(obt[2], 10 + sort ) )
    
    
def test_direct_unionize():

    data = {'savu': np.arange(1000)}
    im = {1: (280, 700), 2: (700, 1000)}


    with mock.patch('allensdk.internal.mouse_connectivity.interval_unionize.'
                    'interval_unionizer.IntervalUnionizer.sort_data_arrays', 
                    new=lambda s, x: x):

        class IU(IntervalUnionizer):
            def extract_data(self, d, l, h, **k):
                return d['savu'][l:h].sum()

        iu = IU()
                        
        iu.interval_map = im
        
        obt = iu.direct_unionize(data)
        
        assert( obt[1] == np.arange(280, 700).sum() )
        assert( obt[2] == np.arange(700, 1000).sum() )
                                    
                        

