import sys
import pandas as pd
import pytest
import numpy as np
import pytz
import datetime
import uuid
import os
import json

from allensdk.test_utilities.custom_comparators import WhitespaceStrippedString


def pytest_assertrepr_compare(config, op, left, right):
    if isinstance(left, WhitespaceStrippedString) and op == "==":
        if isinstance(right, WhitespaceStrippedString):
            right_compare = right.orig
        else:
            right_compare = right
        return ["Comparing strings with whitespace stripped. ",
                f"{left.orig} != {right_compare}.", "Diff:"] + left.diff


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
    from allensdk.brain_observatory.behavior.image_api import ImageApi
    return ImageApi


@pytest.fixture
def max_projection(image_api):
    return image_api.serialize(np.array([[1, 2], [3, 4]]), [.1, .1], 'mm')


@pytest.fixture
def average_image(max_projection):
    return max_projection


@pytest.fixture
def segmentation_mask_image(max_projection):
    return max_projection


@pytest.fixture
def stimulus_presentations_behavior(stimulus_templates, stimulus_presentations):

    image_sets = ['test1','test1', 'test1', 'test2', 'test2' ]
    stimulus_index_df = pd.DataFrame({'image_set': image_sets,
                                      'image_index': [0] * len(image_sets)},
                                       index=pd.Index(stimulus_presentations['start_time'], dtype=np.float64, name='timestamps'))

    df = stimulus_presentations.merge(stimulus_index_df, left_on='start_time', right_index=True)
    return df[sorted(df.columns)]


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
            "reporter_line": ["Ai93(TITL-GCaMP6f)"],
            "driver_line": ["Camk2a-tTA", "Slc17a7-IRES2-Cre"],
            "LabTracks_ID": 416369,
            "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
            "behavior_session_uuid": uuid.uuid4(),
            "emission_lambda": 1.0,
            "excitation_lambda": 1.0,
            "indicator": 'HW',
            "field_of_view_width": 2,
            "field_of_view_height": 2,
            "rig_name": 'my_device',
            "sex": 'M',
            "age": 'P139',
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
    return pd.DataFrame({'cell_roi_id': [123, 321],
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
                          index=pd.Index([None, None], dtype=int, name='cell_specimen_id'))


@pytest.fixture
def valid_roi_ids(cell_specimen_table):
    return set(cell_specimen_table.loc[cell_specimen_table["valid_roi"], "cell_roi_id"].values.tolist())


@pytest.fixture
def dff_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'cell_roi_id': cell_specimen_table['cell_roi_id'],
                         'dff': [np.ones_like(ophys_timestamps)]},
                         index=cell_specimen_table.index)

@pytest.fixture
def corrected_fluorescence_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'cell_roi_id': cell_specimen_table['cell_roi_id'],
                         'corrected_fluorescence': [np.ones_like(ophys_timestamps)]},
                         index=cell_specimen_table.index)

@pytest.fixture
def motion_correction(ophys_timestamps):
    return pd.DataFrame({'x': np.ones_like(ophys_timestamps),
                         'y': np.ones_like(ophys_timestamps)})


@pytest.fixture
def session_data():

    data = {'ophys_experiment_id': 789359614,
            'surface_2p_pixel_size_um': 0.78125,
            "max_projection_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_a13a.png",
            "sync_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/789220000_sync.h5",
            "rig_name": "CAM2P.5",
            "movie_width": 447,
            "movie_height": 512,
            "container_id": 814796558,
            "targeted_structure": "VISp",
            "targeted_depth": 375,
            "stimulus_name": "Unknown",
            "date_of_acquisition": '2018-11-30 23:28:37',
            "reporter_line": ["Ai93(TITL-GCaMP6f)"],
            "driver_line": ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
            "external_specimen_name": 416369,
            "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
            "behavior_stimulus_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl",
            "dff_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/789359614_dff.h5",
            "ophys_cell_segmentation_run_id": 789410052,
            "cell_specimen_table_dict": pd.read_json(os.path.join("/allen", "aibs", "informatics", "nileg", "module_test_data", 'cell_specimen_table_789359614.json'), 'r'), # TODO: I can't write to /allen/aibs/informatics/module_test_data/behavior
            "demix_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/demix/789359614_demixed_traces.h5",
            "average_intensity_projection_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/avgInt_a1X.png",
            "rigid_motion_transform_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/789359614_rigid_motion_transform.csv",
            "segmentation_mask_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_masks.tif",
            "sex": "F",
            "age": "P139"}

    return data
