import math
import mock
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pynwb
import pytest

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi)
from allensdk.test.brain_observatory.behavior.test_eye_tracking_processing import (  # noqa: E501
    create_refined_eye_tracking_df)

from allensdk.brain_observatory.behavior.write_nwb.__main__ import \
    write_behavior_ophys_nwb  # noqa: E501


@pytest.fixture
def rig_geometry():
    """Returns mock rig geometry data"""
    return {
        "monitor_position_mm": [1., 2., 3.],
        "monitor_rotation_deg": [4., 5., 6.],
        "camera_position_mm": [7., 8., 9.],
        "camera_rotation_deg": [10., 11., 12.],
        "led_position": [13., 14., 15.],
        "equipment": "test_rig"}


@pytest.fixture
def eye_tracking_data():
    return create_refined_eye_tracking_df(
        np.array([[0.1, 12 * np.pi, 72 * np.pi, 196 * np.pi, False,
                   196 * np.pi, 12 * np.pi, 72 * np.pi,
                   1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                   13., 14., 15.],
                  [0.2, 20 * np.pi, 90 * np.pi, 225 * np.pi, False,
                   225 * np.pi, 20 * np.pi, 90 * np.pi,
                   2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                   14., 15., 16.]])
    )


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_stimulus_timestamps(nwbfile, stimulus_timestamps,
                                 roundtrip, roundtripper):
    nwb.add_stimulus_timestamps(nwbfile, stimulus_timestamps)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    np.testing.assert_array_almost_equal(stimulus_timestamps,
                                         obt.get_stimulus_timestamps())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_trials(nwbfile, roundtrip, roundtripper, trials):
    nwb.add_trials(nwbfile, trials, {})

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(trials, obt.get_trials(), check_dtype=False)


# licks fixture from test.brain_observatory.behavior.conftest
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

    pd.testing.assert_frame_equal(rewards, obt.get_rewards(),
                                  check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_max_projection(nwbfile, roundtrip, roundtripper,
                            max_projection, image_api):
    nwb.add_max_projection(nwbfile, max_projection)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    assert image_api.deserialize(max_projection) == \
           image_api.deserialize(obt.get_max_projection())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_average_image(nwbfile, roundtrip, roundtripper, average_image,
                           image_api):
    nwb.add_average_image(nwbfile, average_image)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    assert image_api.deserialize(average_image) == \
           image_api.deserialize(obt.get_average_projection())


@pytest.mark.parametrize('roundtrip', [True, False])
def test_segmentation_mask_image(nwbfile, roundtrip, roundtripper,
                                 segmentation_mask_image, image_api):
    nwb.add_segmentation_mask_image(nwbfile, segmentation_mask_image)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    assert image_api.deserialize(segmentation_mask_image) == \
           image_api.deserialize(obt.get_segmentation_mask_image())


@pytest.mark.parametrize('test_partial_metadata', [True, False])
@pytest.mark.parametrize('roundtrip', [True, False])
def test_add_partial_metadata(test_partial_metadata, roundtrip, roundtripper,
                              cell_specimen_table,
                              metadata_fixture, partial_metadata_fixture):
    if test_partial_metadata:
        meta = partial_metadata_fixture
    else:
        meta = metadata_fixture

    nwbfile = pynwb.NWBFile(
        session_description='asession',
        identifier='afile',
        session_start_time=meta['date_of_acquisition']
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
def test_add_task_parameters(nwbfile, roundtrip,
                             roundtripper, task_parameters):
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
def test_get_cell_specimen_table(nwbfile, roundtrip, filter_invalid_rois,
                                 valid_roi_ids, roundtripper,
                                 cell_specimen_table, metadata_fixture,
                                 ophys_timestamps):
    nwb.add_metadata(nwbfile, metadata_fixture, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata_fixture)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi,
                           filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(
            nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        cell_specimen_table = \
            cell_specimen_table[
                cell_specimen_table["cell_roi_id"].isin(
                    valid_roi_ids)]

    pd.testing.assert_frame_equal(
        cell_specimen_table,
        obt.get_cell_specimen_table(),
        check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_dff_traces(nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids,
                        roundtripper, dff_traces, cell_specimen_table,
                        metadata_fixture, ophys_timestamps):
    nwb.add_metadata(nwbfile, metadata_fixture, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata_fixture)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi,
                           filter_invalid_rois=filter_invalid_rois)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(
            nwbfile, filter_invalid_rois=filter_invalid_rois)

    if filter_invalid_rois:
        dff_traces = dff_traces[dff_traces["cell_roi_id"].isin(valid_roi_ids)]

    print(dff_traces)

    print(obt.get_dff_traces())

    pd.testing.assert_frame_equal(
        dff_traces, obt.get_dff_traces(), check_dtype=False)


@pytest.mark.parametrize('roundtrip', [True, False])
@pytest.mark.parametrize("filter_invalid_rois", [True, False])
def test_get_corrected_fluorescence_traces(
        nwbfile, roundtrip, filter_invalid_rois, valid_roi_ids, roundtripper,
        dff_traces, corrected_fluorescence_traces, cell_specimen_table,
        metadata_fixture, ophys_timestamps):
    nwb.add_metadata(nwbfile, metadata_fixture, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata_fixture)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_corrected_fluorescence_traces(nwbfile,
                                          corrected_fluorescence_traces)

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
def test_get_motion_correction(nwbfile, roundtrip, roundtripper,
                               motion_correction, ophys_timestamps,
                               metadata_fixture, cell_specimen_table,
                               dff_traces):
    nwb.add_metadata(nwbfile, metadata_fixture, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata_fixture)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)
    nwb.add_motion_correction(nwbfile, motion_correction)

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    pd.testing.assert_frame_equal(
        motion_correction,
        obt.get_motion_correction(),
        check_dtype=False)


@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("expected", [
    ({
        "monitor_position_mm": [1., 2., 3.],
        "monitor_rotation_deg": [4., 5., 6.],
        "camera_position_mm": [7., 8., 9.],
        "camera_rotation_deg": [10., 11., 12.],
        "led_position": [13., 14., 15.],
        "equipment": "test_rig"
    }),
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
    obtained_eye_rig_geometry = obt.get_eye_tracking_rig_geometry()

    assert obtained_eye_rig_geometry == expected


# NOTE: uses fixtures
# 'nwbfile' and 'roundtripper'
# from allensdk/test/brain_observatory/conftest.py
@pytest.mark.parametrize("roundtrip", [True, False])
def test_add_eye_tracking_data_to_nwbfile(
        tmp_path, nwbfile, eye_tracking_data,
        rig_geometry, roundtripper, roundtrip):
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


@pytest.mark.parametrize("roundtrip", [True, False])
def test_add_events(tmp_path, nwbfile, roundtripper, roundtrip,
                    cell_specimen_table, metadata_fixture, dff_traces,
                    ophys_timestamps):
    # Need to add metadata, cell specimen table, dff traces first
    nwb.add_metadata(nwbfile, metadata_fixture, behavior_only=False)
    nwb.add_cell_specimen_table(nwbfile, cell_specimen_table, metadata_fixture)
    nwb.add_dff_traces(nwbfile, dff_traces, ophys_timestamps)

    events = pd.DataFrame({
        'events': [np.array([0., 0., .69]), np.array([.3, 0.0, .2])],
        'filtered_events': [
            np.array([0.0, 0.0, 0.22949295]),
            np.array([0.09977954, 0.08805513, 0.127039049])
        ],
        'lambda': [0., 1.0],
        'noise_std': [.25, .3],
        'cell_roi_id': [123, 321]
    }, index=pd.Index([42, 84], name='cell_specimen_id'))

    api = BehaviorOphysNwbApi.from_nwbfile(nwbfile)
    nwbfile = api.add_events(
        nwbfile=nwbfile,
        events=events
    )

    if roundtrip:
        obt = roundtripper(nwbfile, BehaviorOphysNwbApi)
    else:
        obt = BehaviorOphysNwbApi.from_nwbfile(nwbfile)

    obtained = obt.get_events()

    pd.testing.assert_frame_equal(obtained, events, check_like=True)


def test_write_behavior_ophys_nwb_no_file():
    """
        This function is testing the fail condition of the
        write_behavior_ophys_nwb method. The main functionality of the
        write_behavior_ophys_nwb method occurs in a try block, and in the
        case that an exception is raised there is functionality in the except
        block to check if any partial output exists, and if so rename that
        file to have a .error suffix before raising the previously
        mentioned exception.

        This test is checking the case where that partial output does not
        exist. In this case we still want to have the original exception
        returned and avoid a FileNotFound error.

        To ensure that we enter the except block, a value of None is passed
        for the session_data argument. This will cause a TypeError when
        write_behavior_ophys_nwb tries to subscript this variable. We are
        checking that, even though no partial output exists, we still get
        this TypeError raised.
    """
    with pytest.raises(TypeError):
        write_behavior_ophys_nwb(
            session_data=None,
            nwb_filepath='',
            skip_eye_tracking=True
        )


def test_write_behavior_ophys_nwb_with_file(tmpdir):
    """
        This function is testing the fail condition of the
        write_behavior_ophys_nwb method. The main functionality of the
        write_behavior_ophys_nwb method occurs in a try block, and in the
        case that an exception is raised there is functionality in the except
        block to check if any partial output exists, and if so rename that
        file to have a .error suffix before raising the previously
        mentioned exception.

        This test is checking the case where a partial output file does
        exist. In this case we still want to have the original exception
        returned and avoid a FileNotFound error, but also check that a new
        file with the .error suffix exists.

        To ensure that we enter the except block, a value of None is passed
        for the session_data argument. This will cause a TypeError when
        write_behavior_ophys_nwb tries to subscript this variable. To get the
        partial output file to exist, we simply create a Path object and
        call the .touch method.

        This test also patched the os.remove method to do nothing. This is
        necessary because the write_behavior_nwb method checks for any
        existing output and removes it before running.
    """
    # Create the dummy .nwb file
    fake_nwb_fp = Path(tmpdir) / 'fake_nwb.nwb'
    Path(str(fake_nwb_fp) + '.inprogress').touch()

    def mock_os_remove(fp):
        pass

    # Patch the os.remove method to do nothing
    with mock.patch('os.remove', side_effects=mock_os_remove):
        with pytest.raises(TypeError):
            write_behavior_ophys_nwb(
                session_data=None,
                nwb_filepath=str(fake_nwb_fp),
                skip_eye_tracking=True
            )

            # Check that the new .error file exists, and that we
            # still get the expected exception
            assert Path(str(fake_nwb_fp) + '.error').exists()
