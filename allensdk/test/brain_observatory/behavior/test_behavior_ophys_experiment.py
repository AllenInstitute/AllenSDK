import os
import datetime
import uuid
import pytest
import pandas as pd
import pytz
import numpy as np
from imageio import imread
from unittest.mock import MagicMock

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
from allensdk.brain_observatory.behavior.write_nwb.__main__ import \
    BehaviorOphysJsonApi
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi, BehaviorOphysLimsApi)
from allensdk.brain_observatory.session_api_utils import (
    sessions_are_equal, compare_session_fields)
from allensdk.brain_observatory.stimulus_info import MONITOR_DIMENSIONS


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("get_expected,get_from_session", [
    [
        lambda ssn_data: ssn_data["ophys_experiment_id"],
        lambda ssn: ssn.ophys_experiment_id
    ],
    [
        lambda ssn_data: imread(ssn_data["max_projection_file"]) / 255,
        lambda ssn: ssn.max_projection
    ]
])
def test_session_from_json(tmpdir_factory, session_data, get_expected,
                           get_from_session):
    session = BehaviorOphysExperiment(api=BehaviorOphysJsonApi(session_data))

    expected = get_expected(session_data)
    obtained = get_from_session(session)

    compare_session_fields(expected, obtained)


@pytest.mark.requires_bamboo
def test_nwb_end_to_end(tmpdir_factory):
    # NOTE: old test oeid 789359614 had no cell specimen ids due to not being
    #       part of the 2021 Visual Behavior release set which broke a ton
    #       of things...

    oeid = 795073741
    tmpdir = 'test_nwb_end_to_end'
    nwb_filepath = os.path.join(str(tmpdir_factory.mktemp(tmpdir)),
                                'nwbfile.nwb')

    d1 = BehaviorOphysExperiment.from_lims(oeid)
    BehaviorOphysNwbApi(nwb_filepath).save(d1)

    d2 = BehaviorOphysExperiment(api=BehaviorOphysNwbApi(nwb_filepath))

    assert sessions_are_equal(d1, d2, reraise=True)


@pytest.mark.nightly
def test_visbeh_ophys_data_set():
    ophys_experiment_id = 789359614
    data_set = BehaviorOphysExperiment.from_lims(ophys_experiment_id)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())

    # All sorts of assert relationships:
    assert data_set.api.extractor.get_foraging_id() == \
        str(data_set.api.get_metadata().behavior_session_uuid)

    stimulus_templates = data_set._stimulus_templates
    assert len(stimulus_templates) == 8
    assert stimulus_templates['im000'].warped.shape == MONITOR_DIMENSIONS
    assert stimulus_templates['im000'].unwarped.shape == MONITOR_DIMENSIONS

    assert len(data_set.licks) == 2421 and set(data_set.licks.columns) \
        == set(['timestamps', 'frame'])
    assert len(data_set.rewards) == 85 and set(data_set.rewards.columns) == \
        set(['timestamps', 'volume', 'autorewarded'])
    assert len(data_set.corrected_fluorescence_traces) == 258 and \
        set(data_set.corrected_fluorescence_traces.columns) == \
        set(['cell_roi_id', 'corrected_fluorescence'])
    np.testing.assert_array_almost_equal(data_set.running_speed.timestamps,
                                         data_set.stimulus_timestamps)
    assert len(data_set.cell_specimen_table) == len(data_set.dff_traces)
    assert data_set.average_projection.data.shape == \
        data_set.max_projection.data.shape
    assert set(data_set.motion_correction.columns) == set(['x', 'y'])
    assert len(data_set.trials) == 602

    expected_metadata = {
        'stimulus_frame_rate': 60.0,
        'full_genotype': 'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93('
                         'TITL-GCaMP6f)/wt',
        'ophys_experiment_id': 789359614,
        'behavior_session_id': 789295700,
        'imaging_plane_group_count': 0,
        'ophys_session_id': 789220000,
        'session_type': 'OPHYS_6_images_B',
        'driver_line': ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
        'cre_line': 'Slc17a7-IRES2-Cre',
        'behavior_session_uuid': uuid.UUID(
            '69cdbe09-e62b-4b42-aab1-54b5773dfe78'),
        'date_of_acquisition': pytz.utc.localize(
            datetime.datetime(2018, 11, 30, 23, 28, 37)),
        'ophys_frame_rate': 31.0,
        'imaging_depth': 375,
        'mouse_id': 416369,
        'experiment_container_id': 814796558,
        'targeted_structure': 'VISp',
        'reporter_line': 'Ai93(TITL-GCaMP6f)',
        'emission_lambda': 520.0,
        'excitation_lambda': 910.0,
        'field_of_view_height': 512,
        'field_of_view_width': 447,
        'indicator': 'GCaMP6f',
        'equipment_name': 'CAM2P.5',
        'age_in_days': 139,
        'sex': 'F',
        'imaging_plane_group': None,
        'project_code': 'VisualBehavior'
    }
    assert data_set.metadata == expected_metadata
    assert data_set.task_parameters == {'reward_volume': 0.007,
                                        'stimulus_distribution': u'geometric',
                                        'stimulus_duration_sec': 0.25,
                                        'stimulus': 'images',
                                        'omitted_flash_fraction': 0.05,
                                        'blank_duration_sec': [0.5, 0.5],
                                        'n_stimulus_frames': 69882,
                                        'task': 'change detection',
                                        'response_window_sec': [0.15, 0.75],
                                        'session_type': u'OPHYS_6_images_B',
                                        'auto_reward_volume': 0.005}


@pytest.mark.requires_bamboo
def test_legacy_dff_api():
    ophys_experiment_id = 792813858
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    session = BehaviorOphysExperiment(api)

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
    session = BehaviorOphysExperiment.from_lims(ophys_experiment_id)
    df = session.stimulus_presentations
    assert df['omitted'].sum() == number_omitted


@pytest.mark.skip
@pytest.mark.requires_bamboo
@pytest.mark.parametrize('ophys_experiment_id', [
    pytest.param(789359614),
    pytest.param(792813858)
])
def test_trial_response_window_bounds_reward(ophys_experiment_id):
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    session = BehaviorOphysExperiment(api)
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


@pytest.mark.parametrize(
    "dilation_frames, z_threshold, eye_tracking_start_value", [
        (5, 9, None),
        (1, 2, None),
        (3, 3, pd.DataFrame([5, 6, 7]))
    ])
def test_eye_tracking(dilation_frames, z_threshold, eye_tracking_start_value):
    mock = MagicMock()
    mock.get_eye_tracking.return_value = pd.DataFrame([1, 2, 3])
    session = BehaviorOphysExperiment(
        api=mock,
        eye_tracking_z_threshold=z_threshold,
        eye_tracking_dilation_frames=dilation_frames)

    if eye_tracking_start_value is not None:
        # Tests that eye_tracking can be set
        session.eye_tracking = eye_tracking_start_value
        obtained = session.eye_tracking
        assert obtained.equals(eye_tracking_start_value)
    else:
        obtained = session.eye_tracking
        assert obtained.equals(pd.DataFrame([1, 2, 3]))
        assert session.api.get_eye_tracking.called_with(
            z_threshold=z_threshold,
            dilation_frames=dilation_frames)


@pytest.mark.requires_bamboo
def test_event_detection():
    ophys_experiment_id = 789359614
    session = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id=ophys_experiment_id)
    events = session.events

    assert len(events) > 0

    expected_columns = ['events', 'filtered_events', 'lambda', 'noise_std',
                        'cell_roi_id']
    assert len(events.columns) == len(expected_columns)
    # Assert they contain the same columns
    assert len(set(expected_columns).intersection(events.columns)) == len(
        expected_columns)

    assert events.index.name == 'cell_specimen_id'

    # All events are the same length
    event_length = len(set([len(x) for x in events['events']]))
    assert event_length == 1


@pytest.mark.requires_bamboo
def test_BehaviorOphysExperiment_property_data():
    ophys_experiment_id = 960410026
    dataset = BehaviorOphysExperiment.from_lims(ophys_experiment_id)

    assert dataset.ophys_session_id == 959458018
    assert dataset.ophys_experiment_id == 960410026


def test_behavior_ophys_experiment_list_data_attributes_and_methods():
    # Test that data related methods/attributes/properties for
    # BehaviorOphysExperiment are returned properly.

    # This test will need to be updated if:
    # 1. Data being returned by class has changed
    # 2. Inheritance of class has changed
    expected = {
        'average_projection',
        'behavior_session_id',
        'cell_specimen_table',
        'corrected_fluorescence_traces',
        'dff_traces',
        'events',
        'eye_tracking',
        'eye_tracking_rig_geometry',
        'get_cell_specimen_ids',
        'get_cell_specimen_indices',
        'get_dff_traces',
        'get_performance_metrics',
        'get_reward_rate',
        'get_rolling_performance_df',
        'get_segmentation_mask_image',
        'licks',
        'max_projection',
        'metadata',
        'motion_correction',
        'ophys_experiment_id',
        'ophys_session_id',
        'ophys_timestamps',
        'raw_running_speed',
        'rewards',
        'roi_masks',
        'running_speed',
        'segmentation_mask_image',
        'stimulus_presentations',
        'stimulus_templates',
        'stimulus_timestamps',
        'task_parameters',
        'trials'
    }

    behavior_ophys_experiment = BehaviorOphysExperiment(api=MagicMock())
    obt = behavior_ophys_experiment.list_data_attributes_and_methods()

    assert any(expected ^ set(obt)) is False
