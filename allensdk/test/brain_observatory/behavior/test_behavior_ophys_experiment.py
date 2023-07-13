import datetime
import os
import uuid

import numpy as np
import pytest
import pytz
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment,
)
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata.behavior_metadata.foraging_id import (  # noqa: E501
    ForagingId,
)
from allensdk.brain_observatory.session_api_utils import sessions_are_equal
from allensdk.brain_observatory.stimulus_info import MONITOR_DIMENSIONS
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.internal.api import db_connection_creator
from pynwb import NWBHDF5IO


@pytest.mark.requires_bamboo
def test_nwb_end_to_end(tmpdir_factory):
    # NOTE: old test oeid 789359614 had no cell specimen ids due to not being
    #       part of the 2021 Visual Behavior release set which broke a ton
    #       of things...

    oeid = 795073741
    tmpdir = "test_nwb_end_to_end"
    nwb_filepath = os.path.join(
        str(tmpdir_factory.mktemp(tmpdir)), "nwbfile.nwb"
    )

    d1 = BehaviorOphysExperiment.from_lims(
        oeid,
        load_stimulus_movie=False
    )
    nwbfile = d1.to_nwb()
    with NWBHDF5IO(nwb_filepath, "w") as nwb_file_writer:
        nwb_file_writer.write(nwbfile)

    d2 = BehaviorOphysExperiment.from_nwb(nwbfile=nwbfile)

    assert sessions_are_equal(
        d1, d2, reraise=True, ignore_keys={"metadata": {"project_code"}}
    )


@pytest.mark.nightly
def test_visbeh_ophys_data_set():
    ophys_experiment_id = 789359614
    data_set = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id,
        exclude_invalid_rois=False,
        load_stimulus_movie=False
    )

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print(np.array(row.to_dict()['mask']).sum())

    lims_db = db_connection_creator(
        fallback_credentials=LIMS_DB_CREDENTIAL_MAP
    )
    behavior_session_id = BehaviorSessionId.from_lims(
        db=lims_db,
        ophys_experiment_id=ophys_experiment_id,
        load_stimulus_movie=False
    )

    # All sorts of assert relationships:
    assert (
        ForagingId.from_lims(
            behavior_session_id=behavior_session_id.value,
            lims_db=lims_db,
            load_stimulus_movie=False
        ).value
        == data_set.metadata["behavior_session_uuid"]
    )

    stimulus_templates = data_set.stimulus_templates
    assert len(stimulus_templates) == 8
    assert stimulus_templates.loc["im000"].warped.shape == MONITOR_DIMENSIONS
    assert stimulus_templates.loc["im000"].unwarped.shape == MONITOR_DIMENSIONS

    assert len(data_set.licks) == 2421 and set(data_set.licks.columns) == set(
        ["timestamps", "frame"]
    )
    assert len(data_set.rewards) == 85 and set(
        data_set.rewards.columns
    ) == set(["timestamps", "volume", "auto_rewarded"])
    assert len(data_set.corrected_fluorescence_traces) == 258 and set(
        data_set.corrected_fluorescence_traces.columns
    ) == set(["cell_roi_id", "corrected_fluorescence", "RMSE", "r"])

    monitor_delay = data_set._stimulus_timestamps.monitor_delay

    np.testing.assert_array_almost_equal(
        data_set.running_speed.timestamps,
        data_set.stimulus_timestamps - monitor_delay,
    )

    assert len(data_set.cell_specimen_table) == len(data_set.dff_traces)
    assert (
        data_set.average_projection.data.shape
        == data_set.max_projection.data.shape
    )
    assert set(data_set.motion_correction.columns) == set(["x", "y"])
    assert len(data_set.trials) == 602

    expected_metadata = {
        "stimulus_frame_rate": 60.0,
        "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93("
        "TITL-GCaMP6f)/wt",
        "ophys_experiment_id": 789359614,
        "behavior_session_id": 789295700,
        "imaging_plane_group_count": 0,
        "ophys_session_id": 789220000,
        "session_type": "OPHYS_6_images_B",
        "driver_line": ["Camk2a-tTA", "Slc17a7-IRES2-Cre"],
        "cre_line": "Slc17a7-IRES2-Cre",
        "behavior_session_uuid": uuid.UUID(
            "69cdbe09-e62b-4b42-aab1-54b5773dfe78"
        ),
        "date_of_acquisition": pytz.utc.localize(
            datetime.datetime(2018, 11, 30, 15, 58, 50, 325000)
        ),
        "ophys_frame_rate": 31.0,
        "imaging_depth": 375,
        "targeted_imaging_depth": 375,
        "mouse_id": "416369",
        "ophys_container_id": 814796558,
        "targeted_structure": "VISp",
        "reporter_line": "Ai93(TITL-GCaMP6f)",
        "emission_lambda": 520.0,
        "excitation_lambda": 910.0,
        "field_of_view_height": 512,
        "field_of_view_width": 447,
        "indicator": "GCaMP6f",
        "equipment_name": "CAM2P.5",
        "age_in_days": 115,
        "sex": "F",
        "imaging_plane_group": None,
        "project_code": "VisualBehavior",
    }
    assert data_set.metadata == expected_metadata
    assert data_set.task_parameters == {
        "reward_volume": 0.007,
        "stimulus_distribution": "geometric",
        "stimulus_duration_sec": 0.25,
        "stimulus": "images",
        "omitted_flash_fraction": 0.05,
        "blank_duration_sec": [0.5, 0.5],
        "n_stimulus_frames": 69882,
        "task": "change detection",
        "response_window_sec": [0.15, 0.75],
        "session_type": "OPHYS_6_images_B",
        "auto_reward_volume": 0.005,
    }


@pytest.mark.requires_bamboo
def test_legacy_dff_api():
    ophys_experiment_id = 792813858
    session = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id=ophys_experiment_id,
        load_stimulus_movie=False
    )

    _, dff_array = session.get_dff_traces()
    for csid in session.dff_traces.index.values:
        dff_trace = session.dff_traces.loc[csid]["dff"]
        ind = session.cell_specimen_table.index.get_loc(csid)
        np.testing.assert_array_almost_equal(dff_trace, dff_array[ind, :])

    assert dff_array.shape[0] == session.dff_traces.shape[0]


@pytest.mark.requires_bamboo
@pytest.mark.parametrize(
    "ophys_experiment_id, number_omitted",
    [pytest.param(789359614, 153), pytest.param(792813858, 129)],
)
def test_stimulus_presentations_omitted(ophys_experiment_id, number_omitted):
    session = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id,
        load_stimulus_movie=False
    )
    df = session.stimulus_presentations
    assert df["omitted"].sum() == number_omitted


@pytest.mark.requires_bamboo
def test_event_detection():
    ophys_experiment_id = 789359614
    session = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id=ophys_experiment_id,
        load_stimulus_movie=False
    )
    events = session.events

    assert len(events) > 0

    expected_columns = [
        "events",
        "filtered_events",
        "lambda",
        "noise_std",
        "cell_roi_id",
    ]
    assert len(events.columns) == len(expected_columns)
    # Assert they contain the same columns
    assert len(set(expected_columns).intersection(events.columns)) == len(
        expected_columns
    )

    assert events.index.name == "cell_specimen_id"

    # All events are the same length
    event_length = len(set([len(x) for x in events["events"]]))
    assert event_length == 1


@pytest.mark.requires_bamboo
def test_BehaviorOphysExperiment_property_data():
    ophys_experiment_id = 960410026
    dataset = BehaviorOphysExperiment.from_lims(
        ophys_experiment_id,
        load_stimulus_movie=False
    )

    assert dataset.ophys_session_id == 959458018
    assert dataset.ophys_experiment_id == 960410026


def test_behavior_ophys_experiment_list_data_attributes_and_methods(
    monkeypatch,
):
    # Test that data related methods/attributes/properties for
    # BehaviorOphysExperiment are returned properly.

    # This test will need to be updated if:
    # 1. Data being returned by class has changed
    # 2. Inheritance of class has changed
    expected = {
        "average_projection",
        "behavior_session_id",
        "cell_specimen_table",
        "corrected_fluorescence_traces",
        "demixed_traces",
        "neuropil_traces",
        "dff_traces",
        "events",
        "eye_tracking",
        "eye_tracking_rig_geometry",
        "get_cell_specimen_ids",
        "get_cell_specimen_indices",
        "get_dff_traces",
        "get_performance_metrics",
        "get_reward_rate",
        "get_rolling_performance_df",
        "get_segmentation_mask_image",
        "licks",
        "max_projection",
        "metadata",
        "motion_correction",
        "ophys_experiment_id",
        "ophys_session_id",
        "ophys_timestamps",
        "raw_running_speed",
        "rewards",
        "roi_masks",
        "running_speed",
        "segmentation_mask_image",
        "stimulus_presentations",
        "stimulus_templates",
        'stimulus_fingerprint_movie_template',
        "stimulus_timestamps",
        "task_parameters",
        "trials",
        "update_targeted_imaging_depth",
    }

    def dummy_init(self):
        pass

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorOphysExperiment, "__init__", dummy_init)
        boe = BehaviorOphysExperiment()
        obt = boe.list_data_attributes_and_methods()

    assert any(expected ^ set(obt)) is False


@pytest.mark.requires_bamboo
def test_stim_v_trials_time(behavior_ophys_experiment_fixture):
    """
    Check that the stimulus and trials tables list the same times
    for frame changes. This was a problem in autumn 2021 because

    a) monitor_delay was not being applied to the stimulus timestamps
    b) the stimulus frame changes and trial frame changes were off by
    one index in the StimulusTimestamps array
    """
    exp = behavior_ophys_experiment_fixture

    stim = exp.stimulus_presentations[
        exp.stimulus_presentations["is_change"]
    ].start_time.reset_index(drop=True)

    trials = (
        exp.trials.query("not aborted")
        .query("go or auto_rewarded")["change_time"]
        .reset_index(drop=True)
    )

    delta = np.abs(stim - trials)
    assert delta.max() < 1.0e-6
