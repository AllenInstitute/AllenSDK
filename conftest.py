import os

import matplotlib
matplotlib.use('agg')
import pytest  # noqa: E402
from allensdk.test_utilities.temp_dir import temp_dir  # noqa: E402


@pytest.fixture(scope="function")
def fn_temp_dir(request):
    return temp_dir(request)


@pytest.fixture(scope="module")
def md_temp_dir(request):
    return temp_dir(request)


def pytest_collection_modifyitems(config, items):
    ''' A pytest magic function. This function is called post-collection and gives us a hook for modifying the 
    collected items.
    '''

    skip_api_endpoint_test = pytest.mark.skipif(
        'TEST_API_ENDPOINT' not in os.environ,
        reason='this test requires that an API endpoint be specified (set the TEST_API_ENDPOINT environment variable).'
    )

    for item in items:
        if 'requires_api_endpoint' in item.keywords:
            item.add_marker(skip_api_endpoint_test)