from allensdk.internal.api.queries.pre_release import BrainObservatoryApiPreRelease
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from six import integer_types
import pytest
import os
import numpy as np

@pytest.fixture(scope='function')
def tmpdir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmpdir')
    return fn


@pytest.mark.prerelease
def test_pre_release_get_containers(tmpdir):

    # Values from original boc/api:
    temp_dir_base = os.path.join(str(tmpdir), 'base-api')
    outfile_base = os.path.join(temp_dir_base, 'manifest.json')
    boc_base = BrainObservatoryCache(manifest_file=outfile_base)
    containers_base = boc_base.get_experiment_containers()

    # # For development: print key/val pairs needed to be populated by adapter:
    # for key, val in sorted(containers_base[0].items(), key=lambda x: x[0]):
    #     print key, val
    # raise

    try:
        temp_dir_extended = os.path.join(str(tmpdir), 'extended-api')
        outfile_extended = os.path.join(temp_dir_extended, 'manifest.json')
        boc_extended = BrainObservatoryCache(manifest_file=outfile_extended, api=BrainObservatoryApiPreRelease())
    except TypeError:
        import allensdk
        raise RuntimeError('Allensdk out-of-date; upgrade with "pip install --force --upgrade allensdk"')

    containers_extended = boc_extended.get_experiment_containers()
    
    # # For development: print key/val pairs actually populated by adapter:
    # for key, val in sorted(containers_extended[0].items(), key=lambda x: x[0]):
    #     print key, val
    # raise


    assert len(containers_extended) > 0
    check_key_list = ['failed', 'tags', 'specimen_name', 'imaging_depth', 'donor_name', 'reporter_line', 'targeted_structure', 'cre_line', 'id']
    for key in check_key_list:
        assert key in containers_extended[0]
    assert len(containers_extended[0]) == len(containers_base[0])
    set(containers_extended[0].keys()) == set(containers_base[0].keys())

    id_container_dict = {}
    for c_e in containers_extended:
        curr_id = c_e['id']
        id_container_dict[curr_id] = c_e

    for c_b in containers_base:
        c_e = id_container_dict[c_b['id']]
        for key in c_e:
            if not c_e[key] == c_b[key]:
                print(key, c_e[key], c_b[key])
                raise Exception()


@pytest.mark.prerelease
def test_pre_release_get_experiments(tmpdir):

    # Values from original boc/api:
    temp_dir_base = os.path.join(str(tmpdir), 'base-api')
    outfile_base = os.path.join(temp_dir_base, 'manifest.json')
    boc_base = BrainObservatoryCache(manifest_file=outfile_base)
    experiments_base = boc_base.get_ophys_experiments()

    # # For development: print key/val pairs needed to be populated by adapter:
    # for key, val in sorted(experiments_base[0].items(), key=lambda x: x[0]):
    #     print key, val
    # raise


    try:
        temp_dir_extended = os.path.join(str(tmpdir), 'extended-api')
        outfile_extended = os.path.join(temp_dir_extended, 'manifest.json')
        boc_extended = BrainObservatoryCache(manifest_file=outfile_extended, api=BrainObservatoryApiPreRelease())
    except TypeError:
        import allensdk
        raise RuntimeError('Allensdk out-of-date; upgrade with "pip install --force --upgrade allensdk"')

    experiments_extended = boc_extended.get_ophys_experiments()

    # # For development: print key/val pairs actually populated by adapter:
    # for key, val in sorted(containers_extended[0].items(), key=lambda x: x[0]):
    #     print key, val
    # raise

    check_key_list = ['acquisition_age_days', 'cre_line', 'donor_name', 'experiment_container_id', 'fail_eye_tracking', 'id', 'imaging_depth', 'reporter_line', 'session_type', 'specimen_name', 'targeted_structure']
    for key in check_key_list:
        assert key in experiments_extended[0]

    assert len(experiments_extended) > 0
    assert set(experiments_base[0].keys()) == set(experiments_extended[0].keys())

    id_experiment_dict = {}
    for c_e in experiments_extended:
        curr_id = c_e['id']
        id_experiment_dict[curr_id] = c_e

    for c_b in experiments_base:
        c_e = id_experiment_dict[c_b['id']]
        for key in c_e:
            # assert c_e[key] == c_b[key]
            if not c_e[key] == c_b[key]:
                print(key, c_e[key], c_b[key])
                raise Exception()


@pytest.mark.prerelease
def test_pre_release_get_cell_specimens(tmpdir):
    
    # Values from original boc/api: Useful debugging code below, commented out
    # import warnings
    # warnings.warn('hard coding tmpdir while I dev, because query takes a long time')
    # temp_dir_base = '/home/nicholasc/tmp/base-api'
    temp_dir_base = os.path.join(str(tmpdir), 'base-api')
    outfile_base = os.path.join(temp_dir_base, 'manifest.json')
    boc_base = BrainObservatoryCache(manifest_file=outfile_base)

    cell_specimens_base = boc_base.get_cell_specimens(include_failed=True)

    # # For development: print key/val pairs needed to be populated by adapter:
    # for container in cell_specimens_base:
    #     if container['all_stim'] == True:
    #         for key, val in sorted(container.items(), key=lambda x: x[0]):
    #             print key, val
    #         raise

    try:
        temp_dir_extended = os.path.join(str(tmpdir), 'extended-api')
        outfile_extended = os.path.join(temp_dir_extended, 'manifest.json')
        boc_extended = BrainObservatoryCache(manifest_file=outfile_extended, api=BrainObservatoryApiPreRelease())
    except TypeError:
        import allensdk
        raise RuntimeError('Allensdk out-of-date; upgrade with "pip install --force --upgrade allensdk"')

    cell_specimens_extended = boc_extended.get_cell_specimens(include_failed=True)


    assert len(cell_specimens_extended) > 0
    assert set(cell_specimens_base[0].keys()) == set(cell_specimens_extended[0].keys())

    id_experiment_dict = {}
    for c_b in cell_specimens_base:
        curr_id = c_b['cell_specimen_id']
        id_experiment_dict[curr_id] = c_b

    for c_e in cell_specimens_extended:
        if c_e['cell_specimen_id'] in id_experiment_dict:
            c_b = id_experiment_dict[c_e['cell_specimen_id']] 
            for key in sorted([key2 for key2 in c_b]):
                assert key in c_e
                if not c_e[key] == c_b[key] and not key == 'specimen_id':       # Failure mode 1: specimen_id changed

                    if isinstance(c_b[key], (float, complex) + integer_types) and isinstance(c_e[key], (float, complex) + integer_types):
                        assert np.isclose(c_e[key], c_b[key], 1e-12) # Failure mode 2: floating-point precision
                    elif c_b[key] is None and isinstance(c_e[key], (float, complex) + integer_types):
                        pass
                    else:
                        # assert c_b[key] is None and isinstance(c_e[key], (int, long, float, complex))
                        print(key, c_e[key], c_b[key])
                        raise Exception()
