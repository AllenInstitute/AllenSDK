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
