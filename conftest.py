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

    skip_nightly_test = pytest.mark.skipif(
        os.getenv('TEST_COMPLETE') != 'true',
        reason='this test is either time/memory/compute expensive or it depends on resources internal to the Allen Institute. '\
            'Either way, it does\'nt run by default and must be opted into (it does run in our nightly builds).'
    )

    skip_flaky_test = pytest.mark.skipif(
        (os.getenv('TEST_COMPLETE') != 'true') or (os.getenv('TEST_FLAKY') == 'true'),
        reason='this test does not consistently pass (for instance, because it makes requests that sometimes time out).'\
            'All such tests should be fixed, but in the mean time we\'ve restricted it to run in our nightly build only '\
            'in order to reduce the prevalence of bogus test results.'
    )

    for item in items:
        if 'requires_api_endpoint' in item.keywords:
            item.add_marker(skip_api_endpoint_test)

        if 'nightly' in item.keywords:
            item.add_marker(skip_nightly_test)

        if 'flaky' in item.keywords:
            item.add_marker(skip_flaky_test)