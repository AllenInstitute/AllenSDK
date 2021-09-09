import datetime
import os
import sys
import uuid

import numpy as np
import pandas as pd
import pytest
import pytz

from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_session_uuid import \
    BehaviorSessionUUID
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.stimulus_frame_rate import \
    StimulusFrameRate
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_ophys_metadata import \
    BehaviorOphysMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.experiment_container_id import \
    ExperimentContainerId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.field_of_view_shape import \
    FieldOfViewShape
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.imaging_depth import \
    ImagingDepth
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.imaging_plane import \
    ImagingPlane
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_experiment_metadata import \
    OphysExperimentMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.ophys_session_id import \
    OphysSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.age import \
    Age
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.driver_line import \
    DriverLine
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.full_genotype import \
    FullGenotype
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.mouse_id import \
    MouseId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.reporter_line import \
    ReporterLine
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.sex import \
    Sex
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.subject_metadata import \
    SubjectMetadata
from allensdk.brain_observatory.behavior.stimulus_processing import \
    StimulusTemplateFactory
from allensdk.test_utilities.custom_comparators import WhitespaceStrippedString


def get_resources_dir():
    behavior_dir = os.path.dirname(__file__)
    return os.path.join(behavior_dir, 'resources')


def pytest_assertrepr_compare(config, op, left, right):
    if isinstance(left, WhitespaceStrippedString) and op == "==":
        if isinstance(right, WhitespaceStrippedString):
            right_compare = right.orig
        else:
            right_compare = right
        return ["Comparing strings with whitespace stripped. ",
                f"{left.orig} != {right_compare}.", "Diff:"] + left.diff


def pytest_ignore_collect(path, config):
    ''' The brain_observatory.ecephys submodule uses
    python 3.6 features that may not be backwards compatible!
    '''

    if sys.version_info < (3, 6):
        return True
    return False


@pytest.fixture
def running_acquisition_df_fixture(running_speed):

    v_sig = np.ones_like(running_speed.values)
    v_in = np.ones_like(running_speed.values)
    dx = np.ones_like(running_speed.values)

    return pd.DataFrame({
        'dx': dx,
        'v_sig': v_sig,
        'v_in': v_in,
    }, index=pd.Index(running_speed.timestamps, name='timestamps'))


@pytest.fixture
def stimulus_templates():

    images = [np.zeros((4, 4)), np.ones((4, 4))]

    image_attributes = [{'image_name': 'test1'}, {'image_name': 'test2'}]
    stimulus_templates = StimulusTemplateFactory.from_unprocessed(
        image_set_name='test', image_attributes=image_attributes,
        images=images)
    return stimulus_templates


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
        'd': [np.array([1]),
              np.array([1, 2]),
              np.array([1, 2, 3]),
              np.array([1, 2, 3, 4]),
              np.array([1, 2, 3, 4, 5])],
    }, index=pd.Index(name='trials_id', data=[0, 1, 2, 3, 4]))


@pytest.fixture
def licks():
    return pd.DataFrame({'timestamps': [1., 2., 3.],
                         'frame': [4., 5., 6.]})


@pytest.fixture
def rewards():
    return pd.DataFrame({'volume': [.01, .01, .01],
                         'timestamps': [1., 2., 3.],
                         'autorewarded': [True, False, False]})


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
def stimulus_presentations_behavior(stimulus_templates,
                                    stimulus_presentations):

    image_sets = ['test1', 'test1', 'test1', 'test2', 'test2']
    start_time = stimulus_presentations['start_time']
    stimulus_index_df = pd.DataFrame({'image_set': image_sets,
                                      'image_index': [0] * len(image_sets)},
                                     index=pd.Index(start_time,
                                                    dtype=np.float64,
                                                    name='timestamps'))

    df = stimulus_presentations.merge(stimulus_index_df,
                                      left_on='start_time',
                                      right_index=True)
    return df[sorted(df.columns)]


@pytest.fixture
def behavior_only_metadata_fixture():
    """Fixture that provides mock behavior only session metadata"""
    subject_meta = SubjectMetadata(
        sex=Sex(sex='M'),
        age=Age(age=139),
        reporter_line=ReporterLine(reporter_line="Ai93(TITL-GCaMP6f)"),
        full_genotype=FullGenotype(
            full_genotype="Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;"
                          "Ai93(TITL-GCaMP6f)/wt"),
        driver_line=DriverLine(
            driver_line=["Camk2a-tTA", "Slc17a7-IRES2-Cre"]),
        mouse_id=MouseId(mouse_id=416369)

    )
    behavior_meta = BehaviorMetadata(
        subject_metadata=subject_meta,
        behavior_session_id=BehaviorSessionId(behavior_session_id=4242),
        equipment=Equipment(equipment_name='my_device'),
        stimulus_frame_rate=StimulusFrameRate(stimulus_frame_rate=60.0),
        session_type=SessionType(session_type='Unknown'),
        behavior_session_uuid=BehaviorSessionUUID(
            behavior_session_uuid=uuid.uuid4())
    )
    return behavior_meta


@pytest.fixture
def metadata_fixture():
    """Fixture that passes all possible behavior ophys session metadata"""
    return {
        "behavior_session_id": 777,
        "ophys_session_id": 999,
        "ophys_experiment_id": 1234,
        "experiment_container_id": 5678,
        "ophys_frame_rate": 31.0,
        "stimulus_frame_rate": 60.0,
        "targeted_structure": "VISp",
        "imaging_depth": 375,
        "session_type": 'Unknown',
        "date_of_acquisition": pytz.utc.localize(datetime.datetime.now()),
        "reporter_line": "Ai93(TITL-GCaMP6f)",
        "driver_line": ["Camk2a-tTA", "Slc17a7-IRES2-Cre"],
        "cre_line": "Slc17a7-IRES2-Cre",
        "mouse_id": 416369,
        "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;"
                         "Ai93(TITL-GCaMP6f)/wt",
        "behavior_session_uuid": uuid.uuid4(),
        "emission_lambda": 1.0,
        "excitation_lambda": 1.0,
        "indicator": 'HW',
        "field_of_view_width": 4,
        "field_of_view_height": 4,
        "equipment_name": 'my_device',
        "sex": 'M',
        "age_in_days": 139,
        "imaging_plane_group": None,
        "imaging_plane_group_count": 0
    }


@pytest.fixture
def behavior_ophys_metadata_fixture(
        behavior_only_metadata_fixture) -> BehaviorOphysMetadata:
    ophys_meta = OphysExperimentMetadata(
        ophys_experiment_id=1234,
        ophys_session_id=OphysSessionId(session_id=999),
        experiment_container_id=ExperimentContainerId(
            experiment_container_id=5678),
        imaging_plane=ImagingPlane(
            ophys_frame_rate=31.0,
            targeted_structure='VISp',
            excitation_lambda=1.0
        ),
        field_of_view_shape=FieldOfViewShape(width=4, height=4),
        imaging_depth=ImagingDepth(imaging_depth=375)
    )
    return BehaviorOphysMetadata(
        behavior_metadata=behavior_only_metadata_fixture,
        ophys_metadata=ophys_meta
    )


@pytest.fixture
def task_parameters():

    return {"blank_duration_sec": [0.5, 0.5],
            "stimulus_duration_sec": 6000.0,
            "omitted_flash_fraction": float('nan'),
            "response_window_sec": [0.15, 0.75],
            "reward_volume": 0.007,
            "session_type": "OPHYS_6_images_B",
            "stimulus": "images",
            "stimulus_distribution": "geometric",
            "task": "DoC_untranslated",
            "n_stimulus_frames": 69882,
            "auto_reward_volume": 0.005
            }


@pytest.fixture
def task_parameters_nan_stimulus_duration():

    return {"blank_duration_sec": [0.5, 0.5],
            "stimulus_duration_sec": np.NaN,
            "omitted_flash_fraction": float('nan'),
            "response_window_sec": [0.15, 0.75],
            "reward_volume": 0.007,
            "session_type": "OPHYS_6_images_B",
            "stimulus": "images",
            "stimulus_distribution": "geometric",
            "task": "DoC_untranslated",
            "n_stimulus_frames": 69882,
            "auto_reward_volume": 0.005
            }


@pytest.fixture
def cell_specimen_table():

    return pd.DataFrame({'cell_roi_id': [123, 321],
                         'x': [0, 2],
                         'y': [0, 2],
                         'width': [2, 2],
                         'height': [2, 2],
                         'valid_roi': [True, False],
                         'max_correction_up': [1., 1.],
                         'max_correction_down': [1., 1.],
                         'max_correction_left': [1., 1.],
                         'max_correction_right': [1., 1.],
                         'mask_image_plane': [1, 1],
                         'ophys_cell_segmentation_run_id': [1, 1],
                         'roi_mask': [np.array([[True, True],
                                                [True, False]]),
                                      np.array([[True, True],
                                                [False, True]])]},
                        index=pd.Index([42, 84], dtype=int,
                        name='cell_specimen_id'))


@pytest.fixture
def valid_roi_ids(cell_specimen_table):
    return set(cell_specimen_table.loc[cell_specimen_table["valid_roi"],
                                       "cell_roi_id"].values.tolist())


@pytest.fixture
def dff_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'cell_roi_id': cell_specimen_table['cell_roi_id'],
                         'dff': [np.ones_like(ophys_timestamps)]},
                        index=cell_specimen_table.index)


@pytest.fixture
def corrected_fluorescence_traces(ophys_timestamps, cell_specimen_table):
    return pd.DataFrame({'cell_roi_id': cell_specimen_table['cell_roi_id'],
                         'corrected_fluorescence': [np.ones_like(ophys_timestamps)]},  # noqa: E501
                        index=cell_specimen_table.index)


@pytest.fixture
def motion_correction(ophys_timestamps):
    return pd.DataFrame({'x': np.ones_like(ophys_timestamps),
                         'y': np.ones_like(ophys_timestamps)})


@pytest.fixture
def session_data():

    data = {
        'behavior_session_id': 789295700,
        'ophys_session_id': 789220000,
        'ophys_experiment_id': 789359614,
        'surface_2p_pixel_size_um': 0.78125,
        'foraging_id': '69cdbe09-e62b-4b42-aab1-54b5773dfe78',
        "max_projection_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_a13a.png",  # noqa: E501
        "sync_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/789220000_sync.h5",  # noqa: E501
        "equipment_name": "CAM2P.5",
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
        "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",  # noqa: E501
        "behavior_stimulus_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl",  # noqa: E501
        "dff_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/789359614_dff.h5",  # noqa: E501
        "ophys_cell_segmentation_run_id": 789410052,
        "cell_specimen_table_dict": pd.read_json(os.path.join("/allen", "aibs", "informatics", "nileg", "module_test_data", 'cell_specimen_table_789359614.json'), 'r'),  # TODO: I can't write to /allen/aibs/informatics/module_test_data/behavior  # noqa: E501
        "demix_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/demix/789359614_demixed_traces.h5",  # noqa: E501
        "average_intensity_projection_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/avgInt_a1X.png",  # noqa: E501
        "rigid_motion_transform_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/789359614_rigid_motion_transform.csv",  # noqa: E501
        "segmentation_mask_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_masks.tif",  # noqa: E501
        "sex": "F",
        "age_in_days": 139,
        "imaging_plane_group": None,
        "imaging_plane_group_count": 0
    }

    return data


@pytest.fixture()
def behavior_stimuli_data_fixture(request):
    """
    This fixture mimicks the behavior experiment stimuli data logs and
    allows parameterization for testing
    """
    images_set_log = request.param.get("images_set_log", [
        ('Image', 'im065', 5.809, 0)])
    images_draw_log = request.param.get("images_draw_log", [
        ([0] + [1] * 3 + [0] * 3)
    ])
    grating_set_log = request.param.get("grating_set_log", [
        ('Ori', 90, 3.585, 0)
    ])
    grating_draw_log = request.param.get("grating_draw_log", [
        ([0] + [1] * 3 + [0] * 3)
    ])
    omitted_flash_frame_log = request.param.get("omitted_flash_frame_log", {
        "grating_0": []
    })
    grating_phase = request.param.get("grating_phase", None)
    grating_spatial_frequency = request.param.get("grating_spatial_frequency",
                                                  None)

    has_images = request.param.get("has_images", True)
    has_grating = request.param.get("has_grating", True)

    resources_dir = get_resources_dir()

    image_data = {
        "set_log": images_set_log,
        "draw_log": images_draw_log,
        "image_path": os.path.join(resources_dir,
                                   'stimulus_template',
                                   'input',
                                   'test_image_set.pkl')
    }

    grating_data = {
        "set_log": grating_set_log,
        "draw_log": grating_draw_log,
        "phase": grating_phase,
        "sf": grating_spatial_frequency
    }

    data = {
        "items": {
            "behavior": {
                "stimuli": {},
                "omitted_flash_frame_log": omitted_flash_frame_log
            }
        }
    }

    if has_images:
        data["items"]["behavior"]["stimuli"]["images"] = image_data

    if has_grating:
        data["items"]["behavior"]["stimuli"]["grating"] = grating_data

    return data
