import sys
import pandas as pd
import pytest
import numpy as np
import pytz
import datetime
import uuid

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
                         'dx': dx,
                         'v_sig': v_sig,
                         'v_in': v_in,
                         }, index=pd.Index(running_speed.timestamps, name='timestamps'))


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
    return pd.DataFrame({'volume': [.01, .01, .01], 'autorewarded': [True, False, False]},
                        index=pd.Index(data=[1., 2., 3.], name='timestamps'))


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


@pytest.fixture
def metadata():

    return {"ophys_experiment_id": 1234,
            "experiment_container_id": 5678,
            "ophys_frame_rate": 31.0,
            "stimulus_frame_rate": 60.0,
            "targeted_structure": "VISp",
            "imaging_depth": 375,
            "session_type": 'Unknown',
            "experiment_datetime": pytz.utc.localize(datetime.datetime.now()),
            "reporter_line": "Ai93(TITL-GCaMP6f)",
            "driver_line": ["Camk2a-tTA", "Slc17a7-IRES2-Cre"],
            "LabTracks_ID": 416369,
            "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
            "behavior_session_uuid": uuid.uuid4(),
            "emission_lambda": 1.0,
            "excitation_lambda": 1.0,
            "indicator": 'HW',
            "field_of_view_width": 2,
            "field_of_view_height": 2,
            "device_name": 'my_device',
            }


@pytest.fixture
def task_parameters():

    return {"blank_duration_sec": [0.5, 0.5],
            "stimulus_duration_sec": 6000.0,
            "omitted_flash_fraction": float('nan'),
            "response_window_sec": [0.15, 0.75],
            "reward_volume": 0.007,
            "stage": "OPHYS_6_images_B",
            "stimulus": "images",
            "stimulus_distribution": "geometric",
            "task": "DoC_untranslated",
            "n_stimulus_frames": 69882
            }


@pytest.fixture
def cell_specimen_table():
    return pd.DataFrame({'cell_specimen_id': [None, None],
                         'x': [1, 1],
                         'y': [1, 1],
                         'width': [1, 1],
                         'height': [1, 1],
                         'valid_roi':[True, False],
                         'max_correction_up':[1., 1.],
                         'max_correction_down':[1., 1.],
                         'max_correction_left':[1., 1.],
                         'max_correction_right':[1., 1.],
                         'mask_image_plane':[1, 1],
                         'ophys_cell_segmentation_run_id':[1, 1],
                         'image_mask': [np.array([[True, True], [False, False]]), np.array([[True, True], [False, False]])]},
                          index=pd.Index([123, 321], dtype=int, name='cell_roi_id'))


@pytest.fixture
def dff_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'dff': [np.ones_like(ophys_timestamps)]},
                         index=cell_specimen_table.index)

@pytest.fixture
def corrected_fluorescence_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'corrected_fluorescence': [np.ones_like(ophys_timestamps)]},
                         index=cell_specimen_table.index)

@pytest.fixture
def motion_correction(ophys_timestamps):
    return pd.DataFrame({'x': np.ones_like(ophys_timestamps),
                         'y': np.ones_like(ophys_timestamps)})
