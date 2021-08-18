import math
import mock
from pathlib import Path

import numpy as np
import pandas as pd
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


# TODO need to add segmentation_mask_image
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
