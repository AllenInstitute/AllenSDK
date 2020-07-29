import os

import matplotlib
matplotlib.use('agg')
import pytest  # noqa: E402
import numpy as np  # noqa: E402

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
        (os.getenv('TEST_COMPLETE') != 'true') and (os.getenv('TEST_FLAKY') != 'true'),
        reason='this test does not consistently pass (for instance, because it makes requests that sometimes time out).'\
            'All such tests should be fixed, but in the mean time we\'ve restricted it to run in our nightly build only '\
            'in order to reduce the prevalence of bogus test results.'
    )

    skip_prerelease_test = pytest.mark.skipif(
        os.environ.get('TEST_PRERELEASE') != 'true',
        reason='prerelease tests are only valid if external and internal data expected to align'
    )

    skip_neuron_test = pytest.mark.skipif(
        os.getenv('TEST_NEURON') != 'true',
        reason='this test depends on the NEURON simulation library. This dependency is not straghtforward to build '\
            'and install, so you must opt in to running this test'
    )

    skip_outside_bamboo_test = pytest.mark.skipif(
        os.environ.get('TEST_BAMBOO') != 'true',
        reason='this test depends on the resources only available to Bamboo agents, but are still fast.  If they are slow, mark with nightly'
    )

    for item in items:
        if 'requires_api_endpoint' in item.keywords:
            item.add_marker(skip_api_endpoint_test)

        if 'nightly' in item.keywords:
            item.add_marker(skip_nightly_test)

        if 'todo_flaky' in item.keywords:
            item.add_marker(skip_flaky_test)

        if 'prerelease' in item.keywords:
            item.add_marker(skip_prerelease_test)

        if 'requires_neuron' in item.keywords:
            item.add_marker(skip_neuron_test)

        if 'requires_bamboo' in item.keywords:
            item.add_marker(skip_outside_bamboo_test)


@pytest.fixture()
def behavior_stimuli_time_fixture(request):
    """
    Fixture that allows for parameterization of behavior_stimuli stimuli
    time data.
    """
    timestamp_count = request.param.get("timestamp_count", 22)
    time_step = request.param.get("time_step", 0.016)

    fixture_params = {
        "timestamp_count": timestamp_count,
        "time_step": time_step
    }

    timestamps = np.array([time_step * i for i in range(timestamp_count)])

    return timestamps, fixture_params


@pytest.fixture()
def behavior_stimuli_data_fixture(request):
    """
    This fixture mimicks the behavior experiment stimuli data logs and
    allows parameterization for testing
    """
    images_set_log = request.param.get("images_set_log", [
        ('Image', 'im065', 5.809, 0)])
    images_draw_log = request.param.get("images_draw_log", [
        ([0] + [1]*3 + [0]*3)
    ])
    grating_set_log = request.param.get("grating_set_log", [
        ('Ori', 90, 3.585, 0)
    ])
    grating_draw_log = request.param.get("grating_draw_log", [
        ([0] + [1]*3 + [0]*3)
    ])
    omitted_flash_frame_log = request.param.get("omitted_flash_frame_log", {
        "grating_0": []
    })

    data = {
        "items": {
            "behavior": {
                "stimuli": {
                    "images": {
                        "set_log": images_set_log,
                        "draw_log": images_draw_log
                    },
                    "grating": {
                        "set_log": grating_set_log,
                        "draw_log": grating_draw_log
                    }
                },
                "omitted_flash_frame_log": omitted_flash_frame_log
            }
        }
    }
    return data
