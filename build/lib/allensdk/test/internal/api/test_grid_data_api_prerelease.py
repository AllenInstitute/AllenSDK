import os

import nrrd
import mock
import pytest
import numpy as np
from numpy.testing import assert_raises

from allensdk.config.manifest import Manifest

from allensdk.internal.api.queries.grid_data_api_prerelease \
        import GridDataApiPrerelease, _get_grid_storage_directories

@pytest.fixture
def storage_dirs(fn_temp_dir):
    return {"111" : os.path.join(fn_temp_dir, "111"),
            "222" : os.path.join(fn_temp_dir, "222")}

@pytest.fixture
def query_result(fn_temp_dir):
    return [{b'id' : 111, b'storage_directory' : os.path.join(fn_temp_dir, "111")},
            {b'id' : 222, b'storage_directory' : os.path.join(fn_temp_dir, "222")}]

@pytest.fixture
def grid_data(storage_dirs, fn_temp_dir):
    gda = GridDataApiPrerelease(storage_dirs)
    return gda


# ----------------------------------------------------------------------------
# module level functions
# ----------------------------------------------------------------------------
@pytest.mark.prerelease()
def test_get_grid_storage_directories(storage_dirs, query_result, fn_temp_dir):
    # ------------------------------------------------------------------------
    # test dirs only have grid/ subdirectory
    with mock.patch('allensdk.internal.core.lims_utilities.query',
                    new=lambda a: query_result):
        obtained = _get_grid_storage_directories(GridDataApiPrerelease.GRID_DATA_DIRECTORY)

    assert not obtained

    # ------------------------------------------------------------------------
    # test returns storage_dirs
    for path in storage_dirs.values():
        Manifest.safe_make_parent_dirs(os.path.join(path, 'grid'))

    with mock.patch('allensdk.internal.core.lims_utilities.query',
                    new=lambda a: query_result):
        obtained = _get_grid_storage_directories(GridDataApiPrerelease.GRID_DATA_DIRECTORY)

    for key, value in obtained:
        assert storage_dirs[key] == value


# ----------------------------------------------------------------------------
# GridDataApiPrerelease class
# ----------------------------------------------------------------------------
@pytest.mark.prerelease()
def test_from_file_name(storage_dirs, fn_temp_dir):
    file_name = os.path.join(fn_temp_dir, 'storage_dirs.json')

    with mock.patch('allensdk.internal.api.queries.grid_data_api_prerelease.'
                    '_get_grid_storage_directories',
                    new=lambda a: storage_dirs):
        grid_data = GridDataApiPrerelease.from_file_name(file_name)

    with mock.patch('allensdk.internal.api.queries.grid_data_api_prerelease.'
            '_get_grid_storage_directories') as ggsd:
        grid_data = GridDataApiPrerelease.from_file_name(file_name)

    ggsd.assert_not_called()
    assert os.path.exists(file_name)


@pytest.mark.prerelease()
def test_download_projection_grid_data(grid_data, fn_temp_dir):

    eye = np.eye(100)
    eid = 111
    target = os.path.join(fn_temp_dir, 'target')

    # test invalid experiment id/no grid
    assert_raises(ValueError, grid_data.download_projection_grid_data, target,
                  0, 'projection_density_100.nrrd')
    assert not os.path.exists(target)

    # test valid
    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage',
                    new=lambda a, b, c: nrrd.write(c, eye)):

        grid_data.download_projection_grid_data(target, 111, 'projection_density_100.nrrd')

    assert os.path.exists(target)
