import os
import datetime
import uuid
import math
import pytest
import pandas as pd
import pytz
import numpy as np
from imageio import imread
from unittest.mock import MagicMock

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.write_nwb.__main__ import BehaviorOphysJsonApi
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi, equals, compare_fields
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_api import BehaviorOphysApiBase
from allensdk.brain_observatory.behavior.image_api import ImageApi


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("get_expected,get_from_session", [
    [
        lambda ssn_data: ssn_data["ophys_experiment_id"],
        lambda ssn: ssn.ophys_experiment_id],
    [
        lambda ssn_data: ssn_data["targeted_structure"],
        lambda ssn: ssn.metadata["targeted_structure"]
    ],
    [
        lambda ssn_data: imread(ssn_data["max_projection_file"]) / 255,
        lambda ssn: ssn.get_max_projection()
    ]

])
def test_session_from_json(tmpdir_factory, session_data, get_expected, get_from_session):
    session = BehaviorOphysSession(api=BehaviorOphysJsonApi(session_data))

    expected = get_expected(session_data)
    obtained = get_from_session(session)

    compare_fields(expected, obtained)


@pytest.mark.xfail
@pytest.mark.requires_bamboo
def test_nwb_end_to_end(tmpdir_factory):
    oeid = 789359614
    nwb_filepath = os.path.join(str(tmpdir_factory.mktemp('test_nwb_end_to_end')), 'nwbfile.nwb')

    d1 = BehaviorOphysSession.from_lims(oeid)
    BehaviorOphysNwbApi(nwb_filepath).save(d1)

    d2 = BehaviorOphysSession(api=BehaviorOphysNwbApi(nwb_filepath))
    equals(d1, d2, reraise=True)


@pytest.mark.nightly
def test_visbeh_ophys_data_set():

    ophys_experiment_id = 789359614
    data_set = BehaviorOphysSession.from_lims(ophys_experiment_id)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())

    # All sorts of assert relationships:
    assert data_set.api.get_foraging_id() == str(data_set.api.get_behavior_session_uuid())
    assert list(data_set.stimulus_templates.values())[0].shape == (8, 918, 1174)
    assert len(data_set.licks) == 2421 and list(data_set.licks.columns) == ['time']
    assert len(data_set.rewards) == 85 and list(data_set.rewards.columns) == ['volume', 'autorewarded']
    assert len(data_set.corrected_fluorescence_traces) == 269 and sorted(data_set.corrected_fluorescence_traces.columns) == ['cell_roi_id', 'corrected_fluorescence']
    np.testing.assert_array_almost_equal(data_set.running_speed.timestamps, data_set.stimulus_timestamps)
    assert len(data_set.cell_specimen_table) == len(data_set.dff_traces)
    assert data_set.average_projection.data.shape == data_set.max_projection.data.shape
    assert list(data_set.motion_correction.columns) == ['x', 'y']
    assert len(data_set.trials) == 602

    assert data_set.metadata == {'stimulus_frame_rate': 60.0,
                                 'full_genotype': 'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt',
                                 'ophys_experiment_id': 789359614,
                                 'session_type': 'OPHYS_6_images_B',
                                 'driver_line': ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
                                 'behavior_session_uuid': uuid.UUID('69cdbe09-e62b-4b42-aab1-54b5773dfe78'),
                                 'experiment_datetime': pytz.utc.localize(datetime.datetime(2018, 11, 30, 23, 28, 37)),
                                 'ophys_frame_rate': 31.0,
                                 'imaging_depth': 375,
                                 'LabTracks_ID': 416369,
                                 'experiment_container_id': 814796558,
                                 'targeted_structure': 'VISp',
                                 'reporter_line': ['Ai93(TITL-GCaMP6f)'],
                                 'emission_lambda': 520.0,
                                 'excitation_lambda': 910.0,
                                 'field_of_view_height': 512,
                                 'field_of_view_width': 447,
                                 'indicator': 'GCAMP6f',
                                 'rig_name': 'CAM2P.5',
                                 'age': 'P139',
                                 'sex': 'F'}

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


@pytest.mark.requires_bamboo
def test_legacy_dff_api():

    ophys_experiment_id = 792813858
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    session = BehaviorOphysSession(api)

    _, dff_array = session.get_dff_traces()
    for csid in session.dff_traces.index.values:
        dff_trace = session.dff_traces.loc[csid]['dff']
        ind = session.get_cell_specimen_indices([csid])[0]
        np.testing.assert_array_almost_equal(dff_trace, dff_array[ind, :])

    assert dff_array.shape[0] == session.dff_traces.shape[0]


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('ophys_experiment_id, number_omitted', [
    pytest.param(789359614, 153),
    pytest.param(792813858, 129)
])
def test_stimulus_presentations_omitted(ophys_experiment_id, number_omitted):
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)
    df = session.stimulus_presentations
    assert df['omitted'].sum() == number_omitted


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('ophys_experiment_id', [
    pytest.param(789359614),
    pytest.param(792813858)
])
def test_trial_response_window_bounds_reward(ophys_experiment_id):

    api = BehaviorOphysLimsApi(ophys_experiment_id)
    session = BehaviorOphysSession(api)
    response_window = session.task_parameters['response_window_sec']
    for _, row in session.trials.iterrows():

        lick_times = [(t - row.change_time) for t in row.lick_times]
        if not np.isnan(row.reward_time):

            # monitor delay is incorporated into the trials table change time
            # TODO: where is this set in the session object?
            camstim_change_time = row.change_time - 0.0351

            reward_time = (row.reward_time - camstim_change_time)
            assert response_window[0] < reward_time + 1 / 60
            assert reward_time < response_window[1] + 1 / 60
            if len(session.licks) > 0:
                assert lick_times[0] < reward_time


@pytest.fixture
def cell_specimen_table_api():

    roi_1 = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    roi_2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    class CellSpecimenTableApi(BehaviorOphysApiBase):

        def get_cell_specimen_table(self):
            return pd.DataFrame(
                {
                    "cell_roi_id": [1, 2],
                    "y": [1, 1],
                    "x": [2, 1],
                    "image_mask": [roi_1, roi_2]
                }, index=pd.Index(data=[10, 11], name="cell_specimen_id")
            )

        def get_segmentation_mask_image(self):
            data = roi_1  # useless image data here
            spacing = (1, 1)
            unit = 'index'
            return ImageApi.serialize(data, spacing, unit)
    return CellSpecimenTableApi()


@pytest.mark.parametrize("roi_ids,expected", [
    [
        1,
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    ],
    [
        None,
        np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
    ]
])
def test_get_roi_masks_by_cell_roi_id(roi_ids, expected, cell_specimen_table_api):
    ssn = BehaviorOphysSession(api=cell_specimen_table_api)
    obtained = ssn._get_roi_masks_by_cell_roi_id(roi_ids)
    assert np.allclose(expected, obtained.values)


@pytest.mark.parametrize("cell_specimen_ids,expected", [
    [
        10,
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    ],
    [
        [11, 10],
        np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
    ]
])
def test_get_roi_masks(cell_specimen_ids, expected, cell_specimen_table_api):
    ssn = BehaviorOphysSession(api=cell_specimen_table_api)
    obtained = ssn.get_roi_masks(cell_specimen_ids)
    assert np.allclose(expected, obtained.values)


@pytest.mark.parametrize("dilation_frames, z_threshold, eye_tracking_start_value", [
    (5, 9, None),
    (1, 2, None),
    (3, 3, pd.DataFrame([5, 6, 7]))
])
def test_eye_tracking(dilation_frames, z_threshold, eye_tracking_start_value):
    mock = MagicMock()
    mock.get_eye_tracking.return_value = pd.DataFrame([1, 2, 3])
    session = BehaviorOphysSession(api=mock,
                                   eye_tracking_z_threshold=z_threshold,
                                   eye_tracking_dilation_frames=dilation_frames)

    if eye_tracking_start_value is not None:
        session.eye_tracking = eye_tracking_start_value
        obtained = session.eye_tracking
        assert not session.api.get_eye_tracking.called
        assert obtained.equals(eye_tracking_start_value)
    else:
        obtained = session.eye_tracking
        assert obtained.equals(pd.DataFrame([1, 2, 3]))
        assert session.api.get_eye_tracking.called_with(z_threshold=z_threshold,
                                                        dilation_frames=dilation_frames)
