import math
import warnings

import numpy as np
import pandas as pd
import pynwb
import pytest

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi)
from allensdk.test.brain_observatory.behavior.test_eye_tracking_processing import create_preload_eye_tracking_df, \
    create_refined_eye_tracking_df


@pytest.fixture
def eye_gaze_data():
    """Returns mock eye gaze data"""
    raw_pupil_areas = pd.Series([2., 4.])
    raw_eye_areas = pd.Series([3., 5.])
    raw_screen_coordinates = pd.DataFrame({
        "y": [2., 4.],
        "x": [3., 5.]
    })
    raw_screen_coordinates_spherical = pd.DataFrame({
        "y": [2., 4.],
        "x": [3., 5.]
    })
    new_pupil_areas = pd.Series([2., 4.])
    new_eye_areas = pd.Series([3., 5.])
    new_screen_coordinates = pd.DataFrame({
        "y": [2., 4.],
        "x": [3., 5.]
    })
    new_screen_coordinates_spherical = pd.DataFrame({
        "y": [2., 4.],
        "x": [3., 5.]
    })
    synced_frame_timestamps = pd.Series([.1, .2])

    return dict(
        raw_pupil_areas=raw_pupil_areas,
        raw_eye_areas=raw_eye_areas,
        raw_screen_coordinates=raw_screen_coordinates,
        raw_screen_coordinates_spherical=raw_screen_coordinates_spherical,
        new_pupil_areas=new_pupil_areas,
        new_eye_areas=new_eye_areas,
        new_screen_coordinates=new_screen_coordinates,
        new_screen_coordinates_spherical=new_screen_coordinates_spherical,
        synced_frame_timestamps=synced_frame_timestamps
    )


@pytest.fixture
def rig_geometry():
    """Returns mock rig geometry data"""
    return {"monitor_position_mm": [1., 2., 3.],
     "monitor_rotation_deg": [4., 5., 6.],
     "camera_position_mm": [7., 8., 9.],
     "camera_rotation_deg": [10., 11., 12.],
     "led_position": [13., 14., 15.],
     "equipment": "test_rig"}


@pytest.fixture
def eye_tracking_data():
    return create_refined_eye_tracking_df(
        np.array([[0.1, 12 * np.pi, 72 * np.pi, 196 * np.pi, False,
                   1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.],
                  [0.2, 20 * np.pi, 90 * np.pi, 225 * np.pi, False,
                   2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]])
    )


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_data_dfs_to_nwbfile(nwbfile, running_data_df, roundtrip, roundtripper):
    # Just make it different from running_data_df
    running_data_df_unfiltered = running_data_df.copy()
    running_data_df_unfiltered['speed'] = running_data_df['speed'] * 2

    unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
    nwbfile = nwb.add_running_data_dfs_to_nwbfile(
        nwbfile,
        running_data_df=running_data_df,
        running_data_df_unfiltered=running_data_df_unfiltered,
        unit_dict=unit_dict)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(
        running_data_df, obt.get_running_data_df(lowpass=True))
    pd.testing.assert_frame_equal(
        running_data_df_unfiltered, obt.get_running_data_df(lowpass=False))


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_templates(nwbfile, stimulus_templates, roundtrip, roundtripper):
    for key, val in stimulus_templates.items():
        nwb.add_stimulus_template(nwbfile, val, key)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    stimulus_templates_obt = obt.get_stimulus_templates()
    for key in set(stimulus_templates.keys()).union(set(stimulus_templates_obt.keys())):
        np.testing.assert_array_almost_equal(stimulus_templates[key], stimulus_templates_obt[key])


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_presentations(nwbfile, stimulus_presentations_behavior, stimulus_timestamps, roundtrip, roundtripper, stimulus_templates):
    nwb.add_stimulus_timestamps(nwbfile, stimulus_timestamps)
    nwb.add_stimulus_presentations(nwbfile, stimulus_presentations_behavior)
    for key, val in stimulus_templates.items():
        nwb.add_stimulus_template(nwbfile, val, key)

        # Add index for this template to NWB in-memory object:
        nwb_template = nwbfile.stimulus_template[key]
        curr_stimulus_index = stimulus_presentations_behavior[stimulus_presentations_behavior['image_set'] == nwb_template.name]
        nwb.add_stimulus_index(nwbfile, curr_stimulus_index, nwb_template)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(stimulus_presentations_behavior, obt.get_stimulus_presentations(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_timestamps(nwbfile, stimulus_timestamps, roundtrip, roundtripper):

    nwb.add_stimulus_timestamps(nwbfile, stimulus_timestamps)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    np.testing.assert_array_almost_equal(stimulus_timestamps, obt.get_stimulus_timestamps())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_trials(nwbfile, roundtrip, roundtripper, trials):

    nwb.add_trials(nwbfile, trials, {})

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(trials, obt.get_trials(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_licks(nwbfile, roundtrip, roundtripper, licks):

    nwb.add_licks(nwbfile, licks)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(licks, obt.get_licks(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_rewards(nwbfile, roundtrip, roundtripper, rewards):

    nwb.add_rewards(nwbfile, rewards)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(rewards, obt.get_rewards(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_max_projection(nwbfile, roundtrip, roundtripper, max_projection, image_api):

    nwb.add_max_projection(nwbfile, max_projection)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    assert image_api.deserialize(max_projection) == image_api.deserialize(obt.get_max_projection())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_average_image(nwbfile, roundtrip, roundtripper, average_image, image_api):

    nwb.add_average_image(nwbfile, average_image)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    assert image_api.deserialize(average_image) == image_api.deserialize(obt.get_average_projection())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_segmentation_mask_image(nwbfile, roundtrip, roundtripper, segmentation_mask_image, image_api):

    nwb.add_segmentation_mask_image(nwbfile, segmentation_mask_image)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    assert image_api.deserialize(segmentation_mask_image) == image_api.deserialize(obt.get_segmentation_mask_image())


@pytest.mark.parametrize('test_partial_metadata', [True, False])
@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_partial_metadata(test_partial_metadata, roundtrip, roundtripper,
                              cell_specimen_table, metadata, partial_metadata):

    meta = partial_metadata if test_partial_metadata else metadata
    nwbfile = pynwb.NWBFile(
        session_description='asession',
        identifier='afile',
        session_start_time=meta['experiment_datetime']
    )
    nwb.add_metadata(nwbfile, meta, behavior_only=False)
    if not test_partial_metadata:
        nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, meta)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    if not test_partial_metadata:
        metadata_obt = obt.get_metadata()
    else:
        with warnings.catch_warnings(record=True) as record:
            metadata_obt = obt.get_metadata()
        exp_warn_msg = "Could not locate 'ophys' module in NWB"
        print(record)

        assert record[0].message.args[0].startswith(exp_warn_msg)

    assert len(metadata_obt) == len(meta)
    for key, val in meta.items():
        assert val == metadata_obt[key]


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_task_parameters(nwbfile, roundtrip, roundtripper, task_parameters):

    nwb.add_task_parameters(nwbfile, task_parameters)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    task_parameters_obt = obt.get_task_parameters()

    assert len(task_parameters_obt) == len(task_parameters)
    for key, val in task_parameters.items():
        if key == 'omitted_flash_fraction':
            if math.isnan(val):
                assert math.isnan(task_parameters_obt[key])
            if math.isnan(task_parameters_obt[key]):
                assert math.isnan(val)
        else:
            assert val == task_parameters_obt[key]


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_cell_specimen_table(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, cell_specimen_table, metadata, ophys_timestamps):

    nwb.add_metadata(nwbfile, metadata, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        cell_specimen_table = cell_specimen_table[cell_specimen_table["cell_roi_id"].isin(valid_roi_ids)]

    print(cell_specimen_table)
    print(obt.get_cell_specimen_table())

    pd.testing.assert_frame_equal(cell_specimen_table, obt.get_cell_specimen_table(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_dff_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, dff_traces, cell_specimen_table, metadata, ophys_timestamps):

    nwb.add_metadata(nwbfile, metadata, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        dff_traces = dff_traces[dff_traces["cell_roi_id"].isin(valid_roi_ids)]

    print(dff_traces)

    print(obt.get_dff_traces())

    pd.testing.assert_frame_equal(dff_traces, obt.get_dff_traces(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_corrected_fluorescence_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, dff_traces, corrected_fluorescence_traces, cell_specimen_table, metadata, ophys_timestamps):

    nwb.add_metadata(nwbfile, metadata, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_corrected_fluorescence_traces(nwbfile, corrected_fluorescence_traces)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi,
                          filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(
            nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        corrected_fluorescence_traces = corrected_fluorescence_traces[
            corrected_fluorescence_traces["cell_roi_id"].isin(valid_roi_ids)]

    print(corrected_fluorescence_traces)
    print(obt.get_corrected_fluorescence_traces())

    pd.testing.assert_frame_equal(
        corrected_fluorescence_traces,
        obt.get_corrected_fluorescence_traces(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_get_motion_correction(nwbfile, roundtrip, roundtripper, motion_correction, ophys_timestamps, metadata, cell_specimen_table, dff_traces):

    nwb.add_metadata(nwbfile, metadata, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_motion_correction(nwbfile, motion_correction)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(motion_correction, obt.get_motion_correction(), check_dtype=False)

@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("expected", [
    ({"geometry": pd.DataFrame({"monitor_position_mm": [1., 2., 3.],
                                "monitor_rotation_deg": [4., 5., 6.],
                                "camera_position_mm": [7., 8., 9.],
                                "camera_rotation_deg": [10., 11., 12.],
                                "led_position_mm": [13., 14., 15.]},
                               index=["x", "y", "z"]),
      "equipment": "test_rig"}),
])
def test_add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile, roundtripper,
                                                       roundtrip,
                                                       rig_geometry,
                                                       expected):
    api = BehaviorOphysNwbApi.from_nwbfile(nwbfile)
    nwbfile = api.add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile,
                                                                rig_geometry)
    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)
    obtained_metadata = obt.get_rig_metadata()

    pd.testing.assert_frame_equal(obtained_metadata["geometry"], expected["geometry"], check_like=True)
    assert obtained_metadata["equipment"] == expected["equipment"]


@pytest.mark.parametrize("roundtrip", [True, False])
def test_add_eye_tracking_data_to_nwbfile(tmp_path, nwbfile, eye_tracking_data, rig_geometry,
                                          roundtripper, roundtrip):
    api = BehaviorOphysNwbApi.from_nwbfile(nwbfile)
    nwbfile = api.add_eye_tracking_data_to_nwb(
        nwbfile=nwbfile,
        eye_tracking_df=eye_tracking_data,
        eye_tracking_rig_geometry=rig_geometry
    )

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    obtained = obt.get_eye_tracking()

    pd.testing.assert_frame_equal(obtained,
                                  eye_tracking_data, check_like=True)
