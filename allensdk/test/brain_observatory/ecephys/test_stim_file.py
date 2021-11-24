import pickle
import operator as op
import os

import pytest
import numpy as np

from allensdk.brain_observatory.ecephys.file_io import stim_file as stim_file


# ideally these would be fixtures, but I want to parametrize over them
def stim_pkl_data():
    return {
        'fps': 1000,
        'pre_blank_sec': 20,
        'stimuli': [{'a': 1}, {'a': 1}],
        'items': {
            'foraging': {
                'encoders': [
                    {
                        'dx': [1, 2, 3]
                    }
                ]
            }
        }
    }


def stim_pkl_data_toplevel_dx():
    return {
        'fps': 1000,
        'pre_blank_sec': 20,
        'dx': [1, 2, 3],
        'stimuli': [{'a': 1}, {'a': 1}],
        'items': {
            'foraging': {
                'encoders': []
            }
        }
    }


@pytest.fixture(params=[stim_pkl_data, stim_pkl_data_toplevel_dx])
def stim_pkl_on_disk(tmpdir_factory, request):
    tmpdir = str(tmpdir_factory.mktemp('stim_files'))
    file_path = os.path.join(tmpdir, 'stim.pkl')

    with open(file_path, 'wb') as pkl_file:
        pickle.dump(request.param(), pkl_file)

    return file_path


@pytest.fixture
def camstimone_pickle_stim_file(stim_pkl_on_disk):
    return stim_file.CamStimOnePickleStimFile.factory(stim_pkl_on_disk)


@pytest.mark.parametrize('prop_name,expected,comp', [
    ['frames_per_second', 1000, op.eq],
    ['pre_blank_sec', 20, op.eq],
    ['angular_wheel_rotation', [1, 2, 3], np.allclose],
    ['angular_wheel_velocity', [1000, 2000, 3000], np.allclose]
])
def test_properties(camstimone_pickle_stim_file, prop_name, expected, comp):
    obtained = getattr(camstimone_pickle_stim_file, prop_name)
    assert comp(obtained, expected)