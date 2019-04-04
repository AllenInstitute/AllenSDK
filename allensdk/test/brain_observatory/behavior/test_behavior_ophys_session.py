import os
import warnings
import datetime
import uuid
import pytest
import pandas as pd
import numpy as np
import h5py

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi

@pytest.mark.nightly
@pytest.mark.parametrize('oeid1, oeid2, expected', [
    pytest.param(789359614, 789359614, True),
    pytest.param(789359614, 739216204, False)
])
def test_equal(oeid1, oeid2, expected):
    d1 = BehaviorOphysSession(oeid1)
    d2 = BehaviorOphysSession(oeid2)

    assert (d1 == d2) == expected


@pytest.mark.nightly
def test_nwb_end_to_end(tmpdir_factory):
    oeid = 789359614
    nwb_filepath = os.path.join(str(tmpdir_factory.mktemp('test_nwb_end_to_end')), 'nwbfile.nwb')
    
    d1 = BehaviorOphysSession(oeid)
    BehaviorOphysNwbApi(nwb_filepath).save(d1)

    d2 = BehaviorOphysSession(789359614, api=BehaviorOphysNwbApi(nwb_filepath))
    assert d1 == d2


@pytest.mark.nightly
def test_visbeh_ophys_data_set():

    ophys_experiment_id = 789359614
    data_set = BehaviorOphysSession(ophys_experiment_id)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())


    # # All sorts of assert relationships:
    # assert data_set.api.get_foraging_id() == str(data_set.api.get_behavior_session_uuid())
    # assert list(data_set.stimulus_templates.values())[0].shape == (8, 918, 1174)
    # assert len(data_set.licks) == 2432 and list(data_set.licks.columns) == ['time']
    # assert len(data_set.rewards) == 85 and list(data_set.rewards.columns) == ['volume', 'autorewarded']
    # assert len(data_set.corrected_fluorescence_traces) == 269 and sorted(data_set.corrected_fluorescence_traces.columns) == ['corrected_fluorescence']
    # np.testing.assert_array_almost_equal(data_set.running_speed.timestamps, data_set.stimulus_timestamps)
    # assert len(data_set.cell_specimen_table) == len(data_set.dff_traces)
    # assert data_set.average_image.GetSize() == data_set.max_projection.GetSize()
    # assert list(data_set.motion_correction.columns) == ['x', 'y']
    # assert len(data_set.trials) == 602

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
                                 'device_name': 'CAM2P.5'}
    
    assert data_set.task_parameters == {'reward_volume': 0.007,
                                        'stimulus_distribution': u'geometric',
                                        'stimulus_duration_sec': 6.0,
                                        'stimulus': 'images',
                                        'blank_duration_sec': [0.5, 0.5],
                                        'n_stimulus_frames': 69882,
                                        'task': 'DoC_untranslated',
                                        'omitted_flash_fraction': float('nan'),
                                        'response_window_sec': [0.15, 0.75],
                                        'stage': u'OPHYS_6_images_B'}
