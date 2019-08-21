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
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi, equals
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi


@pytest.mark.nightly
@pytest.mark.parametrize('oeid1, oeid2, expected', [
    pytest.param(789359614, 789359614, True),
    pytest.param(789359614, 739216204, False)
])
def test_equal(oeid1, oeid2, expected):
    d1 = BehaviorOphysSession.from_lims(oeid1)
    d2 = BehaviorOphysSession.from_lims(oeid2)

    assert equals(d1, d2) == expected

@pytest.mark.nightly
def test_session_from_json(tmpdir_factory, session_data):
    oeid = 789359614

    d1 = BehaviorOphysSession(api=BehaviorOphysJsonApi(session_data))
    d2 = BehaviorOphysSession.from_lims(oeid)

    assert equals(d1, d2)


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
    assert len(data_set.licks) == 2432 and list(data_set.licks.columns) == ['time']
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
                                 'session_type': 'Unknown',
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
            assert response_window[0] < reward_time + 1/60
            assert reward_time < response_window[1] + 1/60
            if len(session.licks) > 0:
                assert lick_times[0] < reward_time
