import os
import warnings
import datetime
import uuid
import pytest
import pandas as pd
import numpy as np
import h5py

from allensdk.core.visual_behavior_ophys_session import VisualBehaviorOphysSession

def test_equal():
    
    oeid1, oeid2 = 789359614, 736590872
    d1 = VisualBehaviorOphysSession(oeid1)
    d2 = VisualBehaviorOphysSession(oeid1)
    d3 = VisualBehaviorOphysSession(oeid2)

    assert d1 == d2
    assert not d1 == d3


def test_visbeh_ophys_data_set():
    
    ophys_experiment_id = 789359614
    data_set = VisualBehaviorOphysSession(ophys_experiment_id)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print np.array(row.to_dict()['mask']).sum()
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print np.array(row.to_dict()['mask']).sum()


    # All sorts of assert relationships:
    assert data_set.api.get_foraging_id(ophys_experiment_id) == str(data_set.api.get_behavior_session_uuid(ophys_experiment_id))
    assert data_set.stimulus_template.shape == (8, 918, 1174)
    assert len(data_set.extended_dataframe) == 602
    assert len(data_set.licks) == 2421 and list(data_set.licks.columns) == ['frame', 'time']
    assert len(data_set.rewards) == 85 and list(data_set.rewards.columns) == ['frame', 'lickspout', 'time', 'volume']
    assert len(data_set.corrected_fluorescence_traces) == 269 and list(data_set.corrected_fluorescence_traces.columns) == ['corrected_fluorescence', 'roi_id',  'timestamps']
    assert sorted(data_set.stimulus_metadata['image_category'].unique()) == sorted(data_set.stimulus_table['image_category'].unique())
    assert sorted(data_set.stimulus_metadata['image_name'].unique()) == sorted(data_set.stimulus_table['image_name'].unique())
    np.testing.assert_array_almost_equal(data_set.running_speed['time'], data_set.stimulus_timestamps)
    assert len(data_set.cell_roi_ids) == len(data_set.dff_traces)
    assert data_set.ophys_timestamps.shape == data_set.dff_traces.timestamps.values[0].shape
    assert data_set.average_image.shape == data_set.max_projection.shape
    assert data_set.motion_correction['framenumber'].shape == data_set.ophys_timestamps.shape
    assert list(data_set.motion_correction.columns) == ['framenumber', 'x', 'y', 'correlation', 'input_x', 'input_y', 'kalman_x', 'kalman_y', 'algorithm', 'type']
    
    assert data_set.metadata == {'stimulus_frame_rate': 60.0, 
                                 'full_genotype': 'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt', 
                                 'ophys_experiment_id': 789359614, 
                                 'session_type': None, 
                                 'driver_line': ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'], 
                                 'behavior_session_uuid': uuid.UUID('69cdbe09-e62b-4b42-aab1-54b5773dfe78'), 
                                 'experiment_date': datetime.datetime(2018, 11, 30, 23, 28, 37), 
                                 'ophys_frame_rate': 31.0, 
                                 'imaging_depth': 375, 
                                 'LabTracks_ID': '416369', 
                                 'experiment_container_id': 814796558, 
                                 'targeted_structure': 'VISp', 
                                 'reporter_line': 'Ai93(TITL-GCaMP6f)'}
    
    assert data_set.task_parameters == {'reward_volume': 0.007, 
                                        'stimulus_distribution': u'geometric', 
                                        'stimulus_duration': 6000.0, 
                                        'stimulus': 'images', 
                                        'blank_duration': 0.5, 
                                        'n_stimulus_frames': 69882, 
                                        'task': 'DoC_untranslated', 
                                        'omitted_flash_fraction': None, 
                                        'response_window': [0.15, 0.75], 
                                        'stage': u'OPHYS_6_images_B'}

def test_get_trials():

    ophys_experiment_id = 789359614 
    dataset = VisualBehaviorOphysSession(ophys_experiment_id)
    trial_df = dataset.get_trials()
    assert len(trial_df) == 304
    assert list(trial_df.columns) == [u'trial', u'change_time', u'initial_image_name', u'change_image_name',
                                      u'trial_type', u'trial_type_color', u'response', u'response_type',
                                      u'response_window', u'lick_times', u'response_latency', u'rewarded',
                                      u'reward_times', u'reward_volume', u'reward_rate', u'start_time',
                                      u'end_time', u'trial_length', u'mouse_id', u'start_date_time']

    all_trial_df = dataset.get_trials(auto_rewarded=True, aborted=True)
    assert len(all_trial_df) == 602
    assert len(all_trial_df) > len(trial_df)

# def test_visbeh_ophys_data_set_events():
    
#     ophys_experiment_id = 702134928
#     api = VisualBehaviorLimsAPI_hackEvents(event_cache_dir='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/events')
#     data_set = VisualBehaviorOphysSession(ophys_experiment_id, api=api)

#     # Not round-tripped
#     data_set.events