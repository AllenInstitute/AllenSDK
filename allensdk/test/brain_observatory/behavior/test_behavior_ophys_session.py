import os
import warnings
import datetime
import uuid
import math
import pytest
import pandas as pd
import pytz
import numpy as np
import h5py
import SimpleITK as sitk
from pandas.util.testing import assert_frame_equal

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.write_nwb.__main__ import BehaviorOphysJsonApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.core.lazy_property import LazyProperty


def equals(A, B):

    field_set = set()
    for key, val in A.__dict__.items():
        if isinstance(val, LazyProperty):
            field_set.add(key)
    for key, val in B.__dict__.items():
        if isinstance(val, LazyProperty):
            field_set.add(key)

    try:
        for field in sorted(field_set):
            x1, x2 = getattr(A, field), getattr(B, field)
            if isinstance(x1, pd.DataFrame):
                assert_frame_equal(x1, x2)
            elif isinstance(x1, np.ndarray):
                np.testing.assert_array_almost_equal(x1, x2)
            elif isinstance(x1, (list,)):
                assert x1 == x2
            elif isinstance(x1, (sitk.Image,)):
                assert x1.GetSize() == x2.GetSize()
                assert x1 == x2
            elif isinstance(x1, (dict,)):
                for key in set(x1.keys()).union(set(x2.keys())):
                    if isinstance(x1[key], (np.ndarray,)):
                        np.testing.assert_array_almost_equal(x1[key], x2[key])
                    elif isinstance(x1[key], (float,)):
                        if math.isnan(x1[key]) or math.isnan(x2[key]):
                            assert math.isnan(x1[key]) and math.isnan(x2[key])
                        else:
                            assert x1[key] == x2[key]
                    else:
                        assert x1[key] == x2[key]

            else:
                assert x1 == x2

    except NotImplementedError as e:
        A_implements_get_field = hasattr(A.api, getattr(type(A), field).getter_name)
        B_implements_get_field = hasattr(B.api, getattr(type(B), field).getter_name)
        assert A_implements_get_field == B_implements_get_field == False

    except (AssertionError, AttributeError) as e:
        return False

    return True

@pytest.mark.nightly
@pytest.mark.parametrize('oeid1, oeid2, expected', [
    pytest.param(789359614, 789359614, True),
    pytest.param(789359614, 739216204, False)
])
def test_equal(oeid1, oeid2, expected):
    d1 = BehaviorOphysSession.from_LIMS(oeid1)
    d2 = BehaviorOphysSession.from_LIMS(oeid2)

    assert equals(d1, d2) == expected

@pytest.mark.nightly
def test_session_from_json(tmpdir_factory):
    oeid = 789359614

    data = {'ophys_experiment_id': 789359614,
            'surface_2p_pixel_size_um': 0.78125,
            "segmentation_mask_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_a13a.png",
            "sync_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/789220000_sync.h5",
            "rig_name": "CAM2P.5",
            "movie_width": 447,
            "movie_height": 512,
            "container_id": 814796558,
            "targeted_structure": "VISp",
            "targeted_depth": 375,
            "stimulus_name": "Unknown",
            "date_of_acquisition": '2018-11-30 23:28:37',
            "reporter_line": "Ai93(TITL-GCaMP6f)",
            "driver_line": ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
            "external_specimen_name": 416369,
            "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
            "behavior_stimulus_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl",
            "dff_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/789359614_dff.h5",
            "ophys_cell_segmentation_run_id": 789410052,
            "cell_specimen_table": open(os.path.join(os.path.dirname(__file__), 'cell_specimen_table_789359614.json'), 'r').read(),
            "demix_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/demix/789359614_demixed_traces.h5",
            "average_intensity_projection_image": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/avgInt_a1X.png",
            "rigid_motion_transform_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/789359614_rigid_motion_transform.csv",
            "external_specimen_name": 416369,
            }

    d1 = BehaviorOphysSession(api=BehaviorOphysJsonApi(data))
    d2 = BehaviorOphysSession.from_LIMS(oeid)

    # print(d1.metadata)
    # print(d2.metadata)

    assert equals(d1, d2)

@pytest.mark.nightly
def test_nwb_end_to_end(tmpdir_factory):
    oeid = 789359614
    nwb_filepath = os.path.join(str(tmpdir_factory.mktemp('test_nwb_end_to_end')), 'nwbfile.nwb')

    d1 = BehaviorOphysSession.from_LIMS(oeid)
    BehaviorOphysNwbApi(nwb_filepath).save(d1)

    d2 = BehaviorOphysSession(api=BehaviorOphysNwbApi(nwb_filepath))
    assert equals(d1, d2)


@pytest.mark.nightly
def test_visbeh_ophys_data_set():

    ophys_experiment_id = 789359614
    data_set = BehaviorOphysSession.from_LIMS(ophys_experiment_id)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())


    # All sorts of assert relationships:
    assert data_set.api.get_foraging_id() == str(data_set.api.get_behavior_session_uuid())
    assert list(data_set.stimulus_templates.values())[0].shape == (8, 918, 1174)
    assert len(data_set.licks) == 2432 and list(data_set.licks.columns) == ['time']
    assert len(data_set.rewards) == 85 and list(data_set.rewards.columns) == ['volume', 'autorewarded']
    assert len(data_set.corrected_fluorescence_traces) == 269 and sorted(data_set.corrected_fluorescence_traces.columns) == ['corrected_fluorescence']
    np.testing.assert_array_almost_equal(data_set.running_speed.timestamps, data_set.stimulus_timestamps)
    assert len(data_set.cell_specimen_table) == len(data_set.dff_traces)
    assert data_set.average_image.GetSize() == data_set.segmentation_mask_image.GetSize()
    assert list(data_set.motion_correction.columns) == ['x', 'y']
    assert len(data_set.trials) == 602

    assert data_set.metadata == {'stimulus_frame_rate': 60.0,
                                 'full_genotype': 'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt',
                                 'ophys_experiment_id': 789359614,
                                 'session_type': 'Unknown',
                                 'driver_line': ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
                                 'behavior_session_uuid': uuid.UUID('69cdbe09-e62b-4b42-aab1-54b5773dfe78'),
                                 'experiment_datetime': pytz.utc.localize(datetime.datetime(2018, 11, 30, 23, 28, 37)),
                                 'ophys_frame_rate': 31.0,
                                 'imaging_depth': 375,
                                 'LabTracks_ID': 416369,
                                 'experiment_container_id': 814796558,
                                 'targeted_structure': 'VISp',
                                 'reporter_line': 'Ai93(TITL-GCaMP6f)',
                                 'emission_lambda': 520.0,
                                 'excitation_lambda': 910.0,
                                 'field_of_view_height': 512,
                                 'field_of_view_width': 447,
                                 'indicator': 'GCAMP6f',
                                 'rig_name': 'CAM2P.5'}

    assert math.isnan(data_set.task_parameters.pop('omitted_flash_fraction'))
    assert data_set.task_parameters == {'reward_volume': 0.007,
                                        'stimulus_distribution': u'geometric',
                                        'stimulus_duration_sec': 6.0,
                                        'stimulus': 'images',
                                        'blank_duration_sec': [0.5, 0.5],
                                        'n_stimulus_frames': 69882,
                                        'task': 'DoC_untranslated',
                                        'response_window_sec': [0.15, 0.75],
                                        'stage': u'OPHYS_6_images_B'}
