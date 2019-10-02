import os

import mock
import pytest
import nrrd
import numpy as np
import pandas as pd

from allensdk.core import json_utilities

from allensdk.internal.core.mouse_connectivity_cache_prerelease \
        import MouseConnectivityCachePrerelease


@pytest.fixture(scope='function')
def mcc(fn_temp_dir):
    storage_dirs = {"111" : os.path.join(fn_temp_dir, "111"),
                    "222" : os.path.join(fn_temp_dir, "222")}

    file_name = os.path.join(fn_temp_dir, 'storage_directories.json')
    json_utilities.write(file_name, storage_dirs)

    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    return MouseConnectivityCachePrerelease(
            manifest_file=manifest_path, storage_directories_file_name=file_name)

@pytest.fixture
def experiments():
    return [{'id':111,
             'age' : "10 wks",
             'gender' : "M",
             'project_code' : "Connectional Atlas",
             'specimen_name' : "",
             'transgenic_line' : "",
             'workflow_state' : "passed",
             'workflows' : ["2P Serial Imaging"],
             'structure_id' : 184,
             'structure_name' : "Frontal pole, cerebral cortex",
             'structure_abbrev' : "FRP",
             'injection_structures' : [
                 {'id' : 184,
                  'name' : "Frontal pole, cerebral cortex",
                  'abbreviation' : 'FRP'},
                 {'id' : 993,
                  'name' : "Secondary motor area",
                  'abbreviation' : 'MOs'}]}]

@pytest.mark.prerelease
def test_init(mcc, fn_temp_dir):
    manifest_path = os.path.join(fn_temp_dir, 'manifest.json')
    assert os.path.exists(manifest_path)


@pytest.mark.prerelease
def test_get_projection_density(mcc, fn_temp_dir):
    eye = np.eye(100)
    eid = 111
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                        'projection_density_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage',
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_projection_density(eid)

    with mock.patch.object(mcc.api.grid_data_api.api,
                           "retrieve_file_from_storage") as mock_rtrv:
        mcc.get_projection_density(eid)

    mock_rtrv.assert_not_called()
    assert np.allclose(obtained, eye)
    assert os.path.exists(path)


@pytest.mark.prerelease
def test_get_injection_density(mcc, fn_temp_dir):
    eye = np.eye(100)
    eid = 111
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                        'injection_density_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage',
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_density(eid)

    with mock.patch.object(mcc.api.grid_data_api.api,
                           "retrieve_file_from_storage") as mock_rtrv:
        mcc.get_injection_density(eid)

    mock_rtrv.assert_not_called()
    assert np.allclose(obtained, eye)
    assert os.path.exists(path)


@pytest.mark.prerelease
def test_get_injection_fraction(mcc, fn_temp_dir):
    eye = np.eye(100)
    eid = 111
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                        'injection_fraction_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage',
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_injection_fraction(eid)

    with mock.patch.object(mcc.api.grid_data_api.api,
                           "retrieve_file_from_storage") as mock_rtrv:
        mcc.get_injection_fraction(eid)

    mock_rtrv.assert_not_called()
    assert np.allclose(obtained, eye)
    assert os.path.exists(path)


@pytest.mark.prerelease
def test_get_data_mask(mcc, fn_temp_dir):
    eye = np.eye(100)
    eid = 111
    path = os.path.join(fn_temp_dir, 'experiment_{0}'.format(eid),
                        'data_mask_25.nrrd')

    with mock.patch('allensdk.internal.api.api_prerelease.ApiPrerelease.'
                    'retrieve_file_from_storage',
                    new=lambda a, b, c: nrrd.write(c, eye)):
        obtained, _ = mcc.get_data_mask(eid)

    with mock.patch.object(mcc.api.grid_data_api.api,
                           "retrieve_file_from_storage") as mock_rtrv:
        mcc.get_data_mask(eid)

    mock_rtrv.assert_not_called()
    assert np.allclose(obtained, eye)
    assert os.path.exists(path)


@pytest.mark.prerelease
def test_filter_experiments(mcc, experiments):

    # ------------------------------------------------------------------------
    # test cre
    cre = mcc.filter_experiments(experiments, cre=True)
    wt = mcc.filter_experiments(experiments, cre=False)

    assert not cre
    assert len(wt) == 1

    # ------------------------------------------------------------------------
    # test injection_structure_ids
    sid_line = mcc.filter_experiments(experiments, injection_structure_ids=[184, 98])

    assert len(sid_line) == 1

    # ------------------------------------------------------------------------
    # test age
    pass_age = mcc.filter_experiments(experiments, age=['10 wks', '12 wks'])
    fail_age = mcc.filter_experiments(experiments, age=['12 wks'])

    assert len(pass_age) == 1
    assert not fail_age

    # ------------------------------------------------------------------------
    # test gender
    pass_gender = mcc.filter_experiments(experiments, gender=['MALE'])
    fail_gender = mcc.filter_experiments(experiments, gender=['f'])

    assert len(pass_gender) == 1
    assert not fail_gender

    # ------------------------------------------------------------------------
    # test workflow-sate
    pass_ws = mcc.filter_experiments(experiments, workflow_state=['qc', 'passed'])
    fail_ws = mcc.filter_experiments(experiments, workflow_state=['failed'])

    assert len(pass_ws) == 1
    assert not fail_ws

    # ------------------------------------------------------------------------
    # test workflows
    pass_w = mcc.filter_experiments(experiments, workflows=['2P SERial ImaGing'])
    fail_w = mcc.filter_experiments(experiments, workflows=['trans-synaptic'])

    assert len(pass_w) == 1
    assert not fail_w

    # ------------------------------------------------------------------------
    # test project_code
    pass_pc = mcc.filter_experiments(experiments, project_code=['ConNECTIOnal Atlas'])
    fail_pc = mcc.filter_experiments(experiments, project_code=['not a code'])

    assert len(pass_pc) == 1
    assert not fail_pc

    # ------------------------------------------------------------------------
    # test a bunch
    conditions = dict(injection_structure_ids=[184, 98],
                      age=['10 wKS', '12 wks'],
                      gender=['maLE'],
                      workflow_state=['qC', 'pASsed'],
                      workflows=['2p serial imaging'])
    passed = mcc.filter_experiments(experiments, **conditions)

    assert len(passed) == 1
