import pytest
import pathlib
import tempfile
import h5py
import json
import numpy as np


@pytest.fixture
def sync_freq_fixture():
    """
    Frequency of test sync file in Hz
    """
    return 100.0


@pytest.fixture
def sync_sample_fixture():
    """
    numpy array indicating the sample
    """
    rng = np.random.default_rng(22312)
    indexes = np.arange(5, 10000, dtype=int)
    values = rng.choice(indexes, 1000, replace=False)
    values = np.sort(values)
    values = values.astype(np.uint32)
    return values


@pytest.fixture
def line_name_fixture():
    """
    List of line names to use in test sync file
    """
    return ['lineA', '', 'lineB', 'lineC', 'lineD']


@pytest.fixture
def line_to_edges_fixture(
        line_name_fixture,
        sync_sample_fixture):
    """
    A dict mapping line name to lists of rising and falling
    edge indexes in the test sync file (note that the lists
    returned are indexes in sync_sample_fixture, not the actual
    edge sample values we expect to be returned by
    SyncDataset.get_falling_edges or SyncDataset.get_rising_edges
    """
    rng = np.random.default_rng(11235813)
    n_samples = len(sync_sample_fixture)

    # leave first 20 samples blank; these will be filled
    # with an erroneous block of "on" bits so that we can
    # test utilities that only select falling edges
    # after the first rising edge
    #
    # the increment of 3 is to make sure there is a gap
    # between each falling edge and the subsequent
    # rising edge
    indexes = np.arange(20, n_samples, 3, dtype=int)

    result = dict()
    for line_name in line_name_fixture:
        changes = rng.choice(indexes, 50, replace=False)
        changes = np.sort(changes)

        # every line starts out with some random initial
        # samles set to 1
        this_rising = [0, ]
        this_falling = [rng.integers(2, 18)]

        for idx in range(0, len(changes), 2):
            this_rising.append(changes[idx])
            this_falling.append(changes[idx+1])
        result[line_name] = {'rising_idx': np.array(this_rising),
                             'falling_idx': np.array(this_falling)}

    return result


@pytest.fixture
def sync_metadata_fixture(
        sync_freq_fixture,
        line_name_fixture):
    """
    Dict representing 'meta' dataset in test sync file
    """
    metadata = {'ni_daq':
                {'device': 'Dev1',
                 'counter_output_freq': sync_freq_fixture,
                 'sample_rate': sync_freq_fixture,
                 'counter_bits': 32,
                 'event_bits': 32},
                'start_time': '2020-10-07 14:01:17.336502',
                'stop_time': '2020-10-07 16:42:24.177205',
                'line_labels': line_name_fixture,
                'timeouts': [],
                'version': '2.2.1+g1bc7438.b42257',
                'sampling_type': 'frequency',
                'file_version': '1.0.0',
                'line_label_revision': 3,
                'total_samples': 10000}

    return metadata


@pytest.fixture
def sync_file_fixture(
        sync_metadata_fixture,
        line_name_fixture,
        line_to_edges_fixture,
        sync_sample_fixture,
        tmp_path_factory):
    """
    Yields the path to a sync file for testing
    """
    tmpdir = pathlib.Path(tmp_path_factory.mktemp('external_sync_test'))
    sync_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='sync')[1])
    n_samples = len(sync_sample_fixture)
    data = np.zeros((n_samples, 2), dtype=np.uint32)
    data[:, 0] = sync_sample_fixture

    for pwr_of_2, line_name in enumerate(line_name_fixture):
        this_rising = line_to_edges_fixture[line_name]['rising_idx']
        this_falling = line_to_edges_fixture[line_name]['falling_idx']
        for rising, falling in zip(this_rising, this_falling):
            data[rising:falling, 1] += 2**pwr_of_2

    with h5py.File(sync_path, 'w') as out_file:
        out_file.create_dataset(
            'data',
            data=data)
        out_file.create_dataset(
            'meta',
            data=json.dumps(sync_metadata_fixture).encode('utf-8'))

    yield sync_path

    try:
        if sync_path.exists():
            sync_path.unlink()
    except PermissionError:
        pass
