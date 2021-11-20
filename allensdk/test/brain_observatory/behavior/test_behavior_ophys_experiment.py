import os
import datetime
import uuid
import pytest
import pandas as pd
import pytz
import numpy as np
from unittest.mock import create_autospec

from pynwb import NWBHDF5IO

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
from allensdk.brain_observatory.behavior.behavior_session import \
    BehaviorSession
from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_files.eye_tracking_file import \
    EyeTrackingFile
from allensdk.brain_observatory.behavior.data_files\
    .rigid_motion_transform_file import \
    RigidMotionTransformFile
from allensdk.brain_observatory.behavior.data_objects import \
    BehaviorSessionId, StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.cell_specimens\
    .cell_specimens import \
    CellSpecimens
from allensdk.brain_observatory.behavior.data_objects.eye_tracking\
    .eye_tracking_table import \
    EyeTrackingTable
from allensdk.brain_observatory.behavior.data_objects.eye_tracking\
    .rig_geometry import \
    RigGeometry as EyeTrackingRigGeometry
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisitionOphys
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.foraging_id import \
    ForagingId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_ophys_metadata import \
    BehaviorOphysMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.multi_plane_metadata.imaging_plane_group \
    import \
    ImagingPlaneGroup
from allensdk.brain_observatory.behavior.data_objects.projections import \
    Projections
from allensdk.brain_observatory.behavior.data_objects.stimuli.util import \
    calculate_monitor_delay
from allensdk.brain_observatory.behavior.data_objects.timestamps\
    .ophys_timestamps import \
    OphysTimestamps
from allensdk.brain_observatory.session_api_utils import (
    sessions_are_equal)
from allensdk.brain_observatory.stimulus_info import MONITOR_DIMENSIONS
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator


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
    nwbfile = d1.to_nwb()
    with NWBHDF5IO(nwb_filepath, 'w') as nwb_file_writer:
        nwb_file_writer.write(nwbfile)

    d2 = BehaviorOphysExperiment.from_nwb(nwbfile=nwbfile)

    assert sessions_are_equal(d1, d2, reraise=True,
                              ignore_keys={'metadata': {'project_code'}})


@pytest.mark.nightly
def test_visbeh_ophys_data_set():
    ophys_experiment_id = 789359614
    data_set = BehaviorOphysExperiment.from_lims(ophys_experiment_id,
                                                 exclude_invalid_rois=False)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())

    lims_db = db_connection_creator(
        fallback_credentials=LIMS_DB_CREDENTIAL_MAP)
    behavior_session_id = BehaviorSessionId.from_lims(
        db=lims_db, ophys_experiment_id=ophys_experiment_id)

    # All sorts of assert relationships:
    assert ForagingId.from_lims(behavior_session_id=behavior_session_id.value,
                                lims_db=lims_db).value == \
        data_set.metadata['behavior_session_uuid']

    stimulus_templates = data_set.stimulus_templates
    assert len(stimulus_templates) == 8
    assert stimulus_templates.loc['im000'].warped.shape == MONITOR_DIMENSIONS
    assert stimulus_templates.loc['im000'].unwarped.shape == MONITOR_DIMENSIONS

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
    session = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id=ophys_experiment_id)

    _, dff_array = session.get_dff_traces()
    for csid in session.dff_traces.index.values:
        dff_trace = session.dff_traces.loc[csid]['dff']
        ind = session.cell_specimen_table.index.get_loc(csid)
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


@pytest.mark.parametrize(
    "dilation_frames, z_threshold", [
        (5, 9),
        (1, 2)
    ])
def test_eye_tracking(dilation_frames, z_threshold, monkeypatch):
    """A very long test just to test that eye tracking arguments are sent to
    EyeTrackingTable factory method from BehaviorOphysExperiment.from_lims"""
    expected = EyeTrackingTable(eye_tracking=pd.DataFrame([1, 2, 3]))
    EyeTrackingTable_mock = create_autospec(EyeTrackingTable)
    EyeTrackingTable_mock.from_data_file.return_value = expected

    etf = create_autospec(EyeTrackingFile, instance=True)
    sf = create_autospec(SyncFile, instance=True)

    with monkeypatch.context() as ctx:
        ctx.setattr('allensdk.brain_observatory.behavior.'
                    'behavior_ophys_experiment.db_connection_creator',
                    create_autospec(db_connection_creator, instance=True))
        ctx.setattr(
            SyncFile, 'from_lims',
            lambda db, ophys_experiment_id: sf)
        ctx.setattr(
            StimulusTimestamps, 'from_sync_file',
            lambda sync_file: create_autospec(StimulusTimestamps,
                                              instance=True))
        ctx.setattr(
            BehaviorSessionId, 'from_lims',
            lambda db, ophys_experiment_id: create_autospec(BehaviorSessionId,
                                                            instance=True))
        ctx.setattr(
            ImagingPlaneGroup, 'from_lims',
            lambda lims_db, ophys_experiment_id: None)
        ctx.setattr(
            BehaviorOphysMetadata, 'from_lims',
            lambda lims_db, ophys_experiment_id,
            is_multiplane: create_autospec(BehaviorOphysMetadata,
                                           instance=True))
        ctx.setattr('allensdk.brain_observatory.behavior.'
                    'behavior_ophys_experiment.calculate_monitor_delay',
                    create_autospec(calculate_monitor_delay))
        ctx.setattr(
            DateOfAcquisitionOphys, 'from_lims',
            lambda lims_db, ophys_experiment_id: create_autospec(
                DateOfAcquisitionOphys, instance=True))
        ctx.setattr(
            BehaviorSession, 'from_lims',
            lambda lims_db, behavior_session_id,
            stimulus_timestamps, monitor_delay, date_of_acquisition:
            BehaviorSession(
                behavior_session_id=None,
                stimulus_timestamps=None,
                running_acquisition=None,
                raw_running_speed=None,
                running_speed=None,
                licks=None,
                rewards=None,
                stimuli=None,
                task_parameters=None,
                trials=None,
                metadata=None,
                date_of_acquisition=None,
            ))
        ctx.setattr(
            OphysTimestamps, 'from_sync_file',
            lambda sync_file: create_autospec(OphysTimestamps,
                                              instance=True))
        ctx.setattr(
            Projections, 'from_lims',
            lambda lims_db, ophys_experiment_id: create_autospec(
                Projections, instance=True))
        ctx.setattr(
            CellSpecimens, 'from_lims',
            lambda lims_db, ophys_experiment_id, ophys_timestamps,
            segmentation_mask_image_spacing, events_params,
            exclude_invalid_rois: create_autospec(
                     BehaviorSession, instance=True))
        ctx.setattr(
            RigidMotionTransformFile, 'from_lims',
            lambda db, ophys_experiment_id: create_autospec(
                RigidMotionTransformFile, instance=True))
        ctx.setattr(
            EyeTrackingFile, 'from_lims',
            lambda db, ophys_experiment_id: etf)
        ctx.setattr(
            EyeTrackingTable, 'from_data_file',
            lambda data_file, sync_file, z_threshold, dilation_frames:
            EyeTrackingTable_mock.from_data_file(
                data_file=data_file, sync_file=sync_file,
                z_threshold=z_threshold, dilation_frames=dilation_frames))
        ctx.setattr(
            EyeTrackingRigGeometry, 'from_lims',
            lambda lims_db, ophys_experiment_id: create_autospec(
                EyeTrackingRigGeometry, instance=True))
        boe = BehaviorOphysExperiment.from_lims(
            ophys_experiment_id=1, eye_tracking_z_threshold=z_threshold,
            eye_tracking_dilation_frames=dilation_frames)

        obtained = boe.eye_tracking
        assert obtained.equals(expected.value)
        EyeTrackingTable_mock.from_data_file.assert_called_with(
            data_file=etf,
            sync_file=sf,
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


def test_behavior_ophys_experiment_list_data_attributes_and_methods(
        monkeypatch):
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

    def dummy_init(self):
        pass

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysExperiment, '__init__', dummy_init)
        boe = BehaviorOphysExperiment()
        obt = boe.list_data_attributes_and_methods()

    assert any(expected ^ set(obt)) is False
