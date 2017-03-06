import os
import shutil
import itertools as it
import warnings

import mock
import pytest

import numpy as np
import nrrd

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


@pytest.fixture(scope='function')
def test_dir(request):

    tmpfs = os.path.normpath(os.path.join('/', 'dev', 'shm'))
    # would like to check mount type, but that requires system calls
    if os.path.exists(tmpfs) and os.path.ismount(tmpfs):
    
        base_path = tmpfs
    
    else:
    
        base_path = os.path.dirname(__file__)
        
        
    fls = os.listdir(base_path)
    while True:
        dname = ''.join(map(str, np.random.randint(0, 10, 6)))
        if dname not in fls:
            break
            
    specific_path = os.path.join(base_path, 'allensdk_test_' + dname)
    os.makedirs(specific_path)
    
    def fin():
        shutil.rmtree(specific_path)
        if os.path.exists(specific_path):
            warnings.warn('test dir {0} still exists!', UserWarning)
        
    request.addfinalizer(fin)
    
    return specific_path
    

@pytest.fixture(scope='function')    
def mcc(test_dir):

    manifest_path = os.path.join(test_dir, 'manifest.json')
    return MouseConnectivityCache(manifest_file=manifest_path)
    
    
def test_init(mcc, test_dir):

    manifest_path = os.path.join(test_dir, 'manifest.json')
    assert( os.path.exists(manifest_path) )
    
    
def test_get_annotation_volume(mcc, test_dir):

    eye = np.eye(100)
    path = os.path.join(test_dir, 'annotation', 'ccf_2016', 
                        'annotation_25.nrrd')

    mcc.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = mcc.get_annotation_volume()
    
    assert( np.allclose(obtained, eye) ) 
    assert( os.path.exists(path) )
    
    
def test_get_template_volume(mcc, test_dir):

    eye = np.eye(100)
    path = os.path.join(test_dir, 'average_template_25.nrrd')

    mcc.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    obtained, _ = mcc.get_template_volume()
    
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )
    
    
def test_get_projection_density(mcc, test_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(test_dir, 'experiment_{0}'.format(eid), 
                        'projection_density_25.nrrd')
                        
#    mcc.api.retrieve_file_over_http = lambda a, b: nrrd.write(b, eye)
    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_projection_density(eid)
    
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )
    
    
def test_get_injection_density(mcc, test_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(test_dir, 'experiment_{0}'.format(eid), 
                        'injection_density_25.nrrd')
                        
    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_density(eid)
    
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )
    
    
def test_get_injection_fraction(mcc, test_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(test_dir, 'experiment_{0}'.format(eid), 
                        'injection_fraction_25.nrrd')
                            
    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_fraction(eid)
    
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )
    
    
def test_get_data_mask(mcc, test_dir):

    eye = np.eye(100)
    eid = 123456789
    path = os.path.join(test_dir, 'experiment_{0}'.format(eid), 
                        'data_mask_25.nrrd')
                        
    with mock.patch('allensdk.api.queries.grid_data_api.GridDataApi.'
                    'retrieve_file_over_http', 
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_data_mask(eid)
    
    assert( np.allclose(obtained, eye) )            
    assert( os.path.exists(path) )
    
    
def test_get_structure_tree(mcc, test_dir):

    dirty_node = {'id': 0, 'structure_id_path': '/0/', 
                  'color_hex_triplet': '000000', 'acronym': 'rt', 
                  'name': 'root', 'structure_sets':[{'id': 1}, {'id': 4}]}
            
    path = os.path.join(test_dir, 'structures.json')
    
    with mock.patch('allensdk.api.queries.ontologies_api.'
                    'OntologiesApi.model_query', 
                    return_value=dirty_node):
                    
        obtained = mcc.get_structure_tree()
        
    assert(len(obtained.nodes()) == 1)
    assert( os.path.exists(path) )
    
    
#def test_get_ontology(mcc, test_dir):

#    with warnings.catch_warnings(record=True) as c:
#        warnings.simplefilter('always')

#        with mock.patch('allensdk.api.queries.ontologies_api.'
#                        'OntologiesApi.model_query', 
#                        return_value=[1, 2]):
#                
#            mcc.get_ontology()
#    
#            assert(len(c) == 3)
#            
#            
#def test_get_structures(mcc, test_dir):

#    with warnings.catch_warnings(record=True) as c:
#        warnings.simplefilter('always')

#        with mock.patch('allensdk.api.queries.ontologies_api.'
#                        'OntologiesApi.model_query', 
#                        return_value=[1, 2]):
#                
#            mcc.get_structures()
#    
#            assert(len(c) == 1)
#            
#            
#def test_get_experiments(mcc, test_dir):

#    experiments = [{'num-voxels': 100, 'injection-volume': 99, 'sum': 98, 
#                    'name': 'foo', 'transgenic-line': 'most_creish', 
#                    'structure-id': 97,}]
#    file_path = os.path.join(test_dir, 'experiments.json')
#    
#    mcc.api.service_query = lambda a, b: experiments    
#    obtained = mcc.get_experiments()
#    
#    assert( os.path.exists(file_path) )
#    assert( 'num_voxels' not in obtained[0] )
    
    
#def test_filter_experiments(mcc, test_dir):


