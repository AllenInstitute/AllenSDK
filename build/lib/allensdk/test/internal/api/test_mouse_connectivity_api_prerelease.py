import os

import nrrd
import mock
import pytest
import numpy as np
from numpy.testing import assert_raises

from allensdk.core import json_utilities

from allensdk.internal.api.queries.mouse_connectivity_api_prerelease \
        import MouseConnectivityApiPrerelease, _experiment_dict

@pytest.fixture
def storage_dirs(fn_temp_dir):
    return {"111" : os.path.join(fn_temp_dir, "111")}

@pytest.fixture
def connectivity(storage_dirs, fn_temp_dir):
    file_name = os.path.join(fn_temp_dir, 'storage_directories.json')
    json_utilities.write(file_name, storage_dirs)

    mca = MouseConnectivityApiPrerelease(file_name)
    return mca

_STRUCTURE_TREE_ROOT_ID = 997
_STRUCTURE_TREE_ROOT_NAME = "root"
_STRUCTURE_TREE_ROOT_ACRONYM = "root"

# ----------------------------------------------------------------------------
# module level functions
# ----------------------------------------------------------------------------
@pytest.mark.prerelease()
def tests_experiment_dict():
    # ------------------------------------------------------------------------
    # null row
    row = {b'id':1,
           b'age' : None,
           b'gender' : None,
           b'project_code' : None,
           b'specimen_name' : None,
           b'transgenic_line' : None,
           b'workflow_state' : None,
           b'workflows' : None,
           b'structure_id' : None,
           b'structure_name' : None,
           b'structure_acronym' : None,
           b'injection_structures_id' : None,
           b'injection_structures_name' : None,
           b'injection_structures_acronym' : None}

    exp = _experiment_dict(row)

    assert exp.pop('id') == 1

    assert exp.get('structure_id') == exp.get('injection_structures')[0].get('id')
    assert exp.get('structure_name') == exp.get('injection_structures')[0].get('name')
    assert exp.get('structure_abbrev') == exp.get('injection_structures')[0].get('abbreviation')

    assert exp.pop('structure_id') == _STRUCTURE_TREE_ROOT_ID
    assert exp.pop('structure_name') == _STRUCTURE_TREE_ROOT_NAME
    assert exp.pop('structure_abbrev') == _STRUCTURE_TREE_ROOT_ACRONYM

    assert len(exp.pop('injection_structures')) == 1

    assert exp.get('workflows')[0] == ""
    assert len(exp.pop('workflows')) == 1

    for value in exp.values():
        assert value == ""


# ----------------------------------------------------------------------------
# MouseConnectivityApiPrerelease class
# ----------------------------------------------------------------------------
@pytest.mark.prerelease()
def test_get_structure_unionizes(connectivity):
    assert_raises(NotImplementedError, connectivity.get_structure_unionizes)


@pytest.mark.prerelease()
def test_download_injection_density(connectivity, storage_dirs, fn_temp_dir):
    eid = 111
    store = storage_dirs[str(eid)]
    source = os.path.join(store, 'grid', 'injection_density_25.nrrd')
    target = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                          'injection_density_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage') as gda:
        connectivity.download_injection_density(target, eid, 25)

    gda.assert_called_once_with(source, target)


@pytest.mark.prerelease()
def test_download_projection_density(connectivity, storage_dirs, fn_temp_dir):
    eid = 111
    store = storage_dirs[str(eid)]
    source = os.path.join(store, 'grid', 'projection_density_25.nrrd')
    target = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                          'projection_density_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage') as gda:
        connectivity.download_projection_density(target, eid, 25)

    gda.assert_called_once_with(source, target)


@pytest.mark.prerelease()
def test_download_injection_fraction(connectivity, storage_dirs, fn_temp_dir):
    eid = 111
    store = storage_dirs[str(eid)]
    source = os.path.join(store, 'grid', 'injection_fraction_25.nrrd')
    target = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                          'injection_fraction_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage') as gda:
        connectivity.download_injection_fraction(target, eid, 25)

    gda.assert_called_once_with(source, target)


@pytest.mark.prerelease()
def test_download_data_mask(connectivity, storage_dirs, fn_temp_dir):
    eid = 111
    store = storage_dirs[str(eid)]
    source = os.path.join(store, 'grid', 'data_mask_25.nrrd')
    target = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                          'data_mask_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage') as gda:
        connectivity.download_data_mask(target, eid, 25)

    gda.assert_called_once_with(source, target)
