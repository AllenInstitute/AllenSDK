import pytest
import pandas as pd
import numpy as np
import math

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema


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


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_metadata(nwbfile, roundtrip, roundtripper, metadata):

    nwb.add_metadata(nwbfile, metadata)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    metadata_obt = obt.get_metadata()

    assert len(metadata_obt) == len(metadata)
    for key, val in metadata.items():
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

    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        cell_specimen_table = cell_specimen_table[cell_specimen_table["cell_roi_id"].isin(valid_roi_ids)]

    pd.testing.assert_frame_equal(cell_specimen_table, obt.get_cell_specimen_table(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_dff_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, dff_traces, cell_specimen_table, metadata, ophys_timestamps):

    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table)
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
def test_get_corrected_fluorescence_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper, dff_traces, corrected_fluorescence_traces, cell_specimen_table, metadata, ophys_timestamps):

    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_corrected_fluorescence_traces(nwbfile, corrected_fluorescence_traces)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi, filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        corrected_fluorescence_traces = corrected_fluorescence_traces[corrected_fluorescence_traces["cell_roi_id"].isin(valid_roi_ids)]

    pd.testing.assert_frame_equal(corrected_fluorescence_traces, obt.get_corrected_fluorescence_traces(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_get_motion_correction(nwbfile, roundtrip, roundtripper, motion_correction, ophys_timestamps, metadata, cell_specimen_table, dff_traces):

    nwb.add_metadata(nwbfile, metadata)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_motion_correction(nwbfile, motion_correction)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(motion_correction, obt.get_motion_correction(), check_dtype=False)
