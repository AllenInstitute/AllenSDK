import os
import shutil

import numpy as np

import pytest


# This is actually a pytest fixture! See below.
def temp_dir(request):

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
    
fn_temp_dir = pytest.fixture(scope='function')(temp_dir)
md_temp_dir = pytest.fixture(scope='module')(temp_dir)
