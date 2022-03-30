import pytest
import numpy as np
import pathlib
import tempfile
import pandas as pd
import h5py
import json

from allensdk.brain_observatory.behavior.data_files.sync_file import (
    SyncFile)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile,
    ReplayStimulusFile,
    MappingStimulusFile)


@pytest.fixture
def basic_running_stim_file_fixture():
    rng = np.random.default_rng()
    return {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": rng.random((100,)),
                        "vsig": rng.uniform(low=0.0, high=5.1, size=(100,)),
                        "vin": rng.uniform(low=4.9, high=5.0, size=(100,)),
                    }]}}}


@pytest.fixture(scope='session')
def stimulus_file_frame_fixture(
        tmp_path_factory,
        helper_functions):
    """
    Writes some skeletal stimulus files (really only good for getting
    frame counts) to disk. Yields a tuple of dicts

    frame_count_lookup maps the type of file to the number
    of expected frames (type of file is 'behavior', 'mapping',
    or 'replay')

    pkl_path_lookup maps the type of file to the path to the
    temporary pickle file
    """

    tmpdir = tmp_path_factory.mktemp('all_frame_count_test')
    pkl_path_lookup = dict()
    pkl_path_lookup['behavior'] = pathlib.Path(
                          tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pkl_path_lookup['mapping'] = pathlib.Path(
                          tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pkl_path_lookup['replay'] = pathlib.Path(
                          tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    frame_count_lookup = {'behavior': 13, 'mapping': 44, 'replay': 76}

    data = {'items':
            {'behavior':
             {'intervalsms':
              list(range(frame_count_lookup['behavior']-1))}}}
    pd.to_pickle(data, pkl_path_lookup['behavior'])

    for key in ('mapping', 'replay'):
        data = {'intervalsms': list(range(frame_count_lookup[key]-1))}
        pd.to_pickle(data, pkl_path_lookup[key])

    output_pkl_path = dict()
    for key in pkl_path_lookup:
        val = pkl_path_lookup[key]
        val = str(val.resolve().absolute())
        output_pkl_path[key] = val

    yield (frame_count_lookup, output_pkl_path)

    for key in pkl_path_lookup:
        helper_functions.windows_safe_cleanup(
                    file_path=pkl_path_lookup[key])


@pytest.fixture(scope='session')
def merge_data_fixture():
    """
    Return a dict keyed on
    'behavior', 'mapping', 'replay'
    each pointing to dx, speed, v_sig, and v_in data, along
    with a dataframe containing those columns

    'kept_mask' will be boolean mask for timesteps
    with dx != 0.0

    n_timesteps
    """

    output = dict()
    rng = np.random.default_rng(229988)
    n_total = 0
    for key in ('behavior', 'mapping', 'replay'):
        nt = rng.integers(20, 40, 1)[0]
        n_total += nt
        this_entry = dict()
        speed = rng.random(nt)
        dx = rng.random(nt)+0.1
        v_in = rng.random(nt)
        v_sig = rng.random(nt)

        n_skip = rng.integers(nt//4, nt//3, 1)[0]
        skipped = np.sort(rng.choice(np.arange(nt, dtype=int),
                                     n_skip,
                                     replace=False))
        dx[skipped] = 0.0

        df = pd.DataFrame(data={'dx': dx,
                                'v_in': v_in,
                                'v_sig': v_sig,
                                'speed': speed})

        kept_mask = np.ones(nt, dtype=bool)
        kept_mask[skipped] = False

        this_entry['speed'] = speed
        this_entry['dx'] = dx
        this_entry['v_in'] = v_in
        this_entry['v_sig'] = v_sig
        this_entry['dataframe'] = df
        this_entry['kept_mask'] = kept_mask
        output[key] = this_entry

    output['n_timesteps'] = n_total
    return output


# Below here we define a self-consistent set of stimulus datasets for testing

@pytest.fixture(scope='session')
def pkl_tmp_dir_fixture(
        tmp_path_factory):
    """
    A directory where the temporary data files can be written
    """
    tmpdir = pathlib.Path(tmp_path_factory.mktemp(
                               'self_consistent_multi_stim'))
    return tmpdir


@pytest.fixture(scope='session')
def behavior_pkl_fixture(
        pkl_tmp_dir_fixture,
        helper_functions):
    """
    Write a pkl file for behavior data.

    Return a dict with
        path_to_pkl
        dx
        vsig
        vin
        n_frames
    """

    rng = np.random.default_rng(77123412)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(
                        dir=pkl_tmp_dir_fixture,
                        prefix='behavior_',
                        suffix='.pkl')[1])
    n_frames = 77

    n_skip = rng.integers(10, n_frames//2, 1)[0]
    to_skip = rng.choice(np.arange(n_frames, dtype=int),
                         n_skip,
                         replace=False)
    kept_mask = np.ones(n_frames, dtype=bool)
    kept_mask[to_skip] = False

    dx = rng.random((n_frames,))
    dx[to_skip] = 0.0
    vsig = rng.uniform(low=0.0, high=5.1, size=(n_frames,))
    vin = rng.uniform(low=4.9, high=5.0, size=(n_frames,))

    data = {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": dx,
                        "vsig": vsig,
                        "vin": vin,
                    }]}}}

    data['items']['behavior']['intervalsms'] = list(range(n_frames-1))

    pd.to_pickle(data, pkl_path)

    output = {
        'path_to_pkl': str(pkl_path.resolve().absolute()),
        'dx': dx,
        'vsig': vsig,
        'vin': vin,
        'n_frames': n_frames,
        'kept_mask': kept_mask}

    yield output
    helper_functions.windows_safe_cleanup(file_path=pkl_path)


@pytest.fixture(scope='session')
def behavior_stim_file_fixture(behavior_pkl_fixture):
    yield BehaviorStimulusFile.from_json(
        dict_repr={'behavior_stimulus_file':
                   behavior_pkl_fixture['path_to_pkl']})


@pytest.fixture(scope='session')
def mapping_pkl_fixture(
        pkl_tmp_dir_fixture,
        helper_functions):
    """
    Write a pkl file for mapping data.

    Return a dict with
        path_to_pkl
        dx
        vsig
        vin
        n_frames
    """

    rng = np.random.default_rng(442138)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(
                        dir=pkl_tmp_dir_fixture,
                        prefix='mapping_',
                        suffix='.pkl')[1])
    n_frames = 53

    n_skip = rng.integers(10, n_frames//2, 1)[0]
    to_skip = rng.choice(np.arange(n_frames, dtype=int),
                         n_skip,
                         replace=False)
    kept_mask = np.ones(n_frames, dtype=bool)
    kept_mask[to_skip] = False

    dx = rng.random((n_frames,))
    dx[to_skip] = 0.0
    vsig = rng.uniform(low=0.0, high=5.1, size=(n_frames,))
    vin = rng.uniform(low=4.9, high=5.0, size=(n_frames,))

    data = {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": dx,
                        "vsig": vsig,
                        "vin": vin,
                    }]}}}

    data['intervalsms'] = list(range(n_frames-1))

    pd.to_pickle(data, pkl_path)

    output = {
        'path_to_pkl': str(pkl_path.resolve().absolute()),
        'dx': dx,
        'vsig': vsig,
        'vin': vin,
        'n_frames': n_frames,
        'kept_mask': kept_mask}

    yield output
    helper_functions.windows_safe_cleanup(file_path=pkl_path)


@pytest.fixture(scope='session')
def mapping_stim_file_fixture(mapping_pkl_fixture):
    yield MappingStimulusFile.from_json(
         dict_repr={'mapping_stimulus_file':
                    mapping_pkl_fixture['path_to_pkl']})


@pytest.fixture(scope='session')
def replay_pkl_fixture(
        pkl_tmp_dir_fixture,
        helper_functions):
    """
    Write a pkl file for replay data.

    Return a dict with
        path_to_pkl
        dx
        vsig
        vin
        n_frames
    """

    rng = np.random.default_rng(55332211)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(
                        dir=pkl_tmp_dir_fixture,
                        prefix='replay_',
                        suffix='.pkl')[1])
    n_frames = 47

    n_skip = rng.integers(10, n_frames//2, 1)[0]
    to_skip = rng.choice(np.arange(n_frames, dtype=int),
                         n_skip,
                         replace=False)
    kept_mask = np.ones(n_frames, dtype=bool)
    kept_mask[to_skip] = False

    dx = rng.random((n_frames,))
    dx[to_skip] = 0.0
    vsig = rng.uniform(low=0.0, high=5.1, size=(n_frames,))
    vin = rng.uniform(low=4.9, high=5.0, size=(n_frames,))

    data = {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": dx,
                        "vsig": vsig,
                        "vin": vin,
                    }]}}}

    data['intervalsms'] = list(range(n_frames-1))

    pd.to_pickle(data, pkl_path)

    output = {
        'path_to_pkl': str(pkl_path.resolve().absolute()),
        'dx': dx,
        'vsig': vsig,
        'vin': vin,
        'n_frames': n_frames,
        'kept_mask': kept_mask}

    yield output
    helper_functions.windows_safe_cleanup(file_path=pkl_path)


@pytest.fixture(scope='session')
def replay_stim_file_fixture(replay_pkl_fixture):
    yield ReplayStimulusFile.from_json(
        dict_repr={'replay_stimulus_file':
                   replay_pkl_fixture['path_to_pkl']})


@pytest.fixture(scope='session')
def sync_path_fixture(
        behavior_pkl_fixture,
        replay_pkl_fixture,
        mapping_pkl_fixture,
        pkl_tmp_dir_fixture,
        helper_functions):
    """
    path to test sync file that can go with these pickle files
    """

    metadata = {'ni_daq':
                {'device': 'Dev1',
                 'counter_output_freq': 100.0,
                 'sample_rate': 100.0,
                 'counter_bits': 32,
                 'event_bits': 32},
                'start_time': '2020-10-07 14:01:17.336502',
                'stop_time': '2020-10-07 16:42:24.177205',
                'line_labels': ['vsync_stim',
                                'stim_running',
                                'vsync_2p',
                                'lick_sensor',
                                'eye_tracking',
                                'behavior_monitoring',
                                'stim_photodiode'],
                'timeouts': [],
                'version': '2.2.1+g1bc7438.b42257',
                'sampling_type': 'frequency',
                'file_version': '1.0.0',
                'line_label_revision': 3,
                'total_samples': 10000}

    sync_path = pathlib.Path(
                    tempfile.mkstemp(
                        dir=pkl_tmp_dir_fixture,
                        prefix='example_',
                        suffix='.sync')[1])

    nb = behavior_pkl_fixture['n_frames']
    nm = mapping_pkl_fixture['n_frames']
    nr = replay_pkl_fixture['n_frames']
    nframes = nb + nm + nr

    n_lines = 5*nframes
    data = np.zeros((n_lines, 2), dtype=np.uint32)
    data[:, 0] = np.arange(n_lines, dtype=np.uint32)+1

    for i_beh in range(2, 2+3*nb, 3):
        data[i_beh:i_beh+1, 1] += 1
    data[1:i_beh+1, 1] += 2

    for i_map in range(i_beh+3, i_beh+1+3*nm, 3):
        data[i_map:i_map+1, 1] += 1
    data[i_beh+2:i_map+1] += 2

    for i_replay in range(i_map+4, i_map+1+3*nb, 3):
        data[i_replay: i_replay+1, 1] += 1
    data[i_map+4: i_replay+1] += 2

    with h5py.File(sync_path, 'w') as out_file:
        out_file.create_dataset(
            'data', data=data)
        out_file.create_dataset(
            'meta',
            data=json.dumps(metadata).encode('utf-8'))

    yield sync_path

    helper_functions.windows_safe_cleanup(file_path=sync_path)


@pytest.fixture(scope='session')
def sync_file_fixture(sync_path_fixture):
    return SyncFile.from_json(
        dict_repr={'sync_file':
                   str(sync_path_fixture.resolve().absolute())})
