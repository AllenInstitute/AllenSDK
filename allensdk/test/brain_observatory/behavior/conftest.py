import sys
import pandas as pd
import pytest
import numpy as np

from allensdk.brain_observatory.image_api import ImageApi


def pytest_ignore_collect(path, config):
    ''' The brain_observatory.ecephys submodule uses python 3.6 features that may not be backwards compatible!
    '''

    if sys.version_info < (3, 6):
        return True
    return False


@pytest.fixture
def running_data_df(running_speed):

    v_sig = np.ones_like(running_speed.values)
    v_in = np.ones_like(running_speed.values)
    dx = np.ones_like(running_speed.values)

    return pd.DataFrame({'speed': running_speed.values,
                         'v_in': v_in,
                         'v_sig': v_sig,
                         'dx': dx}, index=pd.Index(running_speed.timestamps, name='timestamps'))


@pytest.fixture
def stimulus_templates():

    image_template = np.zeros((3, 4, 5))
    image_template[1, :, :] = image_template[1, :, :] + 1
    image_template[2, :, :] = image_template[2, :, :] + 2

    return {'test1': image_template, 'test2': np.zeros((5, 2, 2))}


@pytest.fixture
def ophys_timestamps():
    return np.array([1., 2., 3.])


@pytest.fixture
def trials():
    return pd.DataFrame({
        'start_time': [1., 2., 4., 5., 6.],
        'stop_time': [2., 4., 5., 6., 8.],
        'a': [0.5, 0.4, 0.3, 0.2, 0.1],
        'b': [[], [1], [2, 2], [3], []],
        'c': ['a', 'bb', 'ccc', 'dddd', 'eeeee'],
        'd': [np.array([1]), np.array([1, 2]), np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])],
    }, index=pd.Index(name='trials_id', data=[0, 1, 2, 3, 4]))


@pytest.fixture
def licks():
    return pd.DataFrame({'time': [1., 2., 3.]})


@pytest.fixture
def rewards():
    return pd.DataFrame({'time': [1., 2., 3.], 'volume': [.01, .01, .01], 'autorewarded': [True, False, False]})


@pytest.fixture
def image_api():
    return ImageApi


@pytest.fixture
def max_projection(image_api):
    return image_api.serialize(np.array([[1, 2], [3, 4]]), [.1, .1], 'mm')


@pytest.fixture
def average_image(max_projection):
    return max_projection


@pytest.fixture
def stimulus_index(stimulus_templates):
    image_sets_list = list(stimulus_templates.keys())
    image_sets = image_sets_list + image_sets_list
    return pd.DataFrame({'image_set': image_sets,
                         'image_index': [0] * len(image_sets)},
                        index=pd.Index(np.arange(len(image_sets), dtype=np.float64), name='timestamps'))
