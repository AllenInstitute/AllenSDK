import os

import pytest


def pytest_ignore_collect(path, config):
    ''' These tests (or the code they test) can only run on the local network at the Allen Institute for Brain Science.
    '''
    return(os.getenv('TEST_COMPLETE') != 'true') and (os.getenv('TEST_INTERNAL') != 'true')
