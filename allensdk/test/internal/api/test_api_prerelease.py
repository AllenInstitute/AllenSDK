import os

import nrrd
import pytest
import numpy as np

from allensdk.internal.api.api_prerelease import ApiPrerelease


@pytest.fixture
def api():
    return ApiPrerelease()


@pytest.mark.prerelease()
def test_retrieve_file_from_storage(api, fn_temp_dir):
    eye = np.eye(100)

    target = os.path.join(fn_temp_dir, 'target')
    store = os.path.join(fn_temp_dir, 'store')
    nrrd.write(store, eye)

    api.retrieve_file_from_storage(store, target)

    assert os.path.exists(target)
