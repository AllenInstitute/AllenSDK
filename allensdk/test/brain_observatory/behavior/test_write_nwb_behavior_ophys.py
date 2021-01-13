import math
import warnings
import numpy as np
import pandas as pd
import pynwb
import pytest

from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi)
import allensdk.brain_observatory.nwb as nwb
from allensdk.test.brain_observatory.behavior.test_eye_tracking_processing import create_preload_eye_tracking_df


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_speed_to_nwbfile(nwbfile, running_speed, roundtrip, roundtripper):
    nwbfile = nwb.add_running_speed_to_nwbfile(nwbfile, running_speed)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    running_speed_obt = obt.get_running_speed()
    assert np.allclose(running_speed.timestamps, running_speed_obt.timestamps)
    assert np.allclose(running_speed.values, running_speed_obt.values)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_running_data_df_to_nwbfile(nwbfile, running_data_df, roundtrip, roundtripper):
    unit_dict = {'v_sig': 'V', 'v_in': 'V', 'speed': 'cm/s', 'timestamps': 's', 'dx': 'cm'}
    nwbfile = nwb.add_running_data_df_to_nwbfile(nwbfile, running_data_df, unit_dict)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(running_data_df, obt.get_running_data_df())


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
def test_add_stimulus_presentations(nwbfile, stimulus_presentations_behavior, stimulus_timestamps, roundtrip, roundtripper,
                                    stimulus_templates):
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
    nwb.add_metadata(nwbfile, meta)
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
def test_get_cell_specimen_table(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, cell_specimen_table,
                                 metadata, ophys_timestamps):
    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        cell_specimen_table = cell_specimen_table[cell_specimen_table["cell_roi_id"].isin(valid_roi_ids)]

    pd.testing.assert_frame_equal(cell_specimen_table, obt.get_cell_specimen_table(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_dff_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, dff_traces, cell_specimen_table,
                        metadata, ophys_timestamps):
    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        dff_traces = dff_traces[dff_traces["cell_roi_id"].isin(valid_roi_ids)]

    pd.testing.assert_frame_equal(dff_traces, obt.get_dff_traces(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_corrected_fluorescence_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, dff_traces,
                                           corrected_fluorescence_traces, cell_specimen_table, metadata, ophys_timestamps):
    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_corrected_fluorescence_traces(nwbfile, corrected_fluorescence_traces)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        corrected_fluorescence_traces = corrected_fluorescence_traces[
            corrected_fluorescence_traces["cell_roi_id"].isin(valid_roi_ids)]

    pd.testing.assert_frame_equal(corrected_fluorescence_traces, obt.get_corrected_fluorescence_traces(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_get_motion_correction(nwbfile, roundtrip, roundtripper, motion_correction, ophys_timestamps, metadata,
                               cell_specimen_table, dff_traces):
    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_motion_correction(nwbfile, motion_correction)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(motion_correction, obt.get_motion_correction(), check_dtype=False)


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("eye_tracking_rig_geometry, expected", [
    ({"monitor_position_mm": [1., 2., 3.],
      "monitor_rotation_deg": [4., 5., 6.],
      "camera_position_mm": [7., 8., 9.],
      "camera_rotation_deg": [10., 11., 12.],
      "led_position": [13., 14., 15.],
      "equipment": "test_rig"},

     #  Expected
     {"geometry": pd.DataFrame({"monitor_position_mm": [1., 2., 3.],
                                "monitor_rotation_deg": [4., 5., 6.],
                                "camera_position_mm": [7., 8., 9.],
                                "camera_rotation_deg": [10., 11., 12.],
                                "led_position_mm": [13., 14., 15.]},
                               index=["x", "y", "z"]),
      "equipment": "test_rig"}),
])
def test_add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile, roundtripper,
                                                       roundtrip,
                                                       eye_tracking_rig_geometry,
                                                       expected):
    nwbfile = nwb.add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile,
                                                                eye_tracking_rig_geometry)
    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)
    obtained_metadata = obt.get_rig_metadata()

    pd.testing.assert_frame_equal(obtained_metadata["geometry"], expected["geometry"], check_like=True)
    assert obtained_metadata["equipment"] == expected["equipment"]


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize(("eye_tracking_frame_times, eye_dlc_tracking_data, "
                          "eye_gaze_data, expected_pupil_data, expected_gaze_data"), [
                             (
                                     # eye_tracking_frame_times
                                     pd.Series([3., 4., 5., 6., 7.]),
                                     # eye_dlc_tracking_data
                                     {"pupil_params": create_preload_eye_tracking_df(np.full((5, 5), 1.)),
                                      "cr_params": create_preload_eye_tracking_df(np.full((5, 5), 2.)),
                                      "eye_params": create_preload_eye_tracking_df(np.full((5, 5), 3.))},
                                     # eye_gaze_data
                                     {"raw_pupil_areas": pd.Series([2., 4., 6., 8., 10.]),
                                      "raw_eye_areas": pd.Series([3., 5., 7., 9., 11.]),
                                      "raw_screen_coordinates": pd.DataFrame(
                                          {"y": [2., 4., 6., 8., 10.], "x": [3., 5., 7., 9., 11.]}),
                                      "raw_screen_coordinates_spherical": pd.DataFrame(
                                          {"y": [2., 4., 6., 8., 10.], "x": [3., 5., 7., 9., 11.]}),
                                      "new_pupil_areas": pd.Series([2., 4., np.nan, 8., 10.]),
                                      "new_eye_areas": pd.Series([3., 5., np.nan, 9., 11.]),
                                      "new_screen_coordinates": pd.DataFrame(
                                          {"y": [2., 4., np.nan, 8., 10.], "x": [3., 5., np.nan, 9., 11.]}),
                                      "new_screen_coordinates_spherical": pd.DataFrame(
                                          {"y": [2., 4., np.nan, 8., 10.], "x": [3., 5., np.nan, 9., 11.]}),
                                      "synced_frame_timestamps": pd.Series([3., 4., 5., 6., 7.])},
                                     # expected_pupil_data
                                     pd.DataFrame({"corneal_reflection_center_x": [2.] * 5,
                                                   "corneal_reflection_center_y": [2.] * 5,
                                                   "corneal_reflection_height": [4.] * 5,
                                                   "corneal_reflection_width": [4.] * 5,
                                                   "corneal_reflection_phi": [2.] * 5,
                                                   "pupil_center_x": [1.] * 5,
                                                   "pupil_center_y": [1.] * 5,
                                                   "pupil_height": [2.] * 5,
                                                   "pupil_width": [2.] * 5,
                                                   "pupil_phi": [1.] * 5,
                                                   "eye_center_x": [3.] * 5,
                                                   "eye_center_y": [3.] * 5,
                                                   "eye_height": [6.] * 5,
                                                   "eye_width": [6.] * 5,
                                                   "eye_phi": [3.] * 5},
                                                  index=[3., 4., 5., 6., 7.]),
                                     # expected_gaze_data
                                     pd.DataFrame({"raw_eye_area": [3., 5., 7., 9., 11.],
                                                   "raw_pupil_area": [2., 4., 6., 8., 10.],
                                                   "raw_screen_coordinates_x_cm": [3., 5., 7., 9., 11.],
                                                   "raw_screen_coordinates_y_cm": [2., 4., 6., 8., 10.],
                                                   "raw_screen_coordinates_spherical_x_deg": [3., 5., 7., 9., 11.],
                                                   "raw_screen_coordinates_spherical_y_deg": [2., 4., 6., 8., 10.],
                                                   "filtered_eye_area": [3., 5., np.nan, 9., 11.],
                                                   "filtered_pupil_area": [2., 4., np.nan, 8., 10.],
                                                   "filtered_screen_coordinates_x_cm": [3., 5., np.nan, 9., 11.],
                                                   "filtered_screen_coordinates_y_cm": [2., 4., np.nan, 8., 10.],
                                                   "filtered_screen_coordinates_spherical_x_deg": [3., 5., np.nan, 9., 11.],
                                                   "filtered_screen_coordinates_spherical_y_deg": [2., 4., np.nan, 8., 10.]},
                                                  index=[3., 4., 5., 6., 7.])
                             ),
                         ])
def test_add_eye_tracking_data_to_nwbfile(nwbfile, roundtripper, roundtrip,
                                          eye_tracking_frame_times,
                                          eye_dlc_tracking_data,
                                          eye_gaze_data,
                                          expected_pupil_data, expected_gaze_data):
    nwbfile = nwb.add_eye_tracking_data_to_nwbfile(nwbfile,
                                                   eye_tracking_frame_times,
                                                   eye_dlc_tracking_data,
                                                   eye_gaze_data)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)
    obtained_pupil_data = obt.get_pupil_data()
    obtained_screen_gaze_data = obt.get_screen_gaze_data(include_filtered_data=True)

    pd.testing.assert_frame_equal(obtained_pupil_data,
                                  expected_pupil_data, check_like=True)
    pd.testing.assert_frame_equal(obtained_screen_gaze_data,
                                  expected_gaze_data, check_like=True)
