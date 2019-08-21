# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import pandas as pd
import pytest
import allensdk.brain_observatory.roi_masks as roi_masks


def test_init_by_pixels():
    a = np.array([[0, 0], [1, 1], [1, 0]])

    m = roi_masks.create_roi_mask(2, 2, [0, 0, 0, 0], pix_list=a)

    mp = m.get_mask_plane()

    assert mp[0, 0] == 1
    assert mp[1, 1] == 1
    assert mp[1, 0] == 0
    assert mp[1, 1] == 1

    assert m.x == 0
    assert m.width == 2
    assert m.y == 0
    assert m.height == 2


def test_init_by_pixels_with_border():
    a = np.array([[1, 1], [2, 1]])

    m = roi_masks.create_roi_mask(3, 3, [1, 1, 1, 1], pix_list=a)

    assert m.x == 1
    assert m.width == 2
    assert m.y == 1
    assert m.height == 1
    assert m.overlaps_motion_border is True


def test_init_by_pixels_large():
    a = np.random.random((512, 512))
    a[a > 0.5] = 1

    m = roi_masks.create_roi_mask(
        512, 512, [0, 0, 0, 0], pix_list=np.argwhere(a))

    npx = len(np.where(a)[0])
    assert npx == len(np.where(m.get_mask_plane())[0])


def test_create_neuropil_mask():

    image_width = 100
    image_height = 80

    # border = [image_width-1, 0, image_height-1, 0]
    border = [5, 5, 5, 5]

    roi_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    roi_mask[40:45, 30:35] = 1

    combined_binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    combined_binary_mask[:, 45:] = 1

    roi = roi_masks.create_roi_mask(image_w=image_width, image_h=image_height, border=border, roi_mask=roi_mask)
    obtained = roi_masks.create_neuropil_mask(roi, border, combined_binary_mask)

    expected_mask = np.zeros((58-27, 45-17), dtype=np.uint8)
    expected_mask[:, :] = 1

    assert np.allclose(expected_mask, obtained.mask)
    assert obtained.x == 17
    assert obtained.y == 27
    assert obtained.width == 28
    assert obtained.height == 31


def test_create_empty_neuropil_mask():
    image_width = 100
    image_height = 80

    # border = [image_width-1, 0, image_height-1, 0]
    border = [5, 5, 5, 5]

    roi_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    roi_mask[40:45, 30:35] = 1

    combined_binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    combined_binary_mask[:, :] = 1

    roi = roi_masks.create_roi_mask(image_w=image_width, image_h=image_height, border=border, roi_mask=roi_mask)
    obtained = roi_masks.create_neuropil_mask(roi, border, combined_binary_mask)

    assert obtained.mask is None
    assert 'zero_pixels' in obtained.flags


@pytest.fixture
def image_dims():
    return {
        'width': 100,
        'height': 100
    }


@pytest.fixture
def motion_border():
    return [5.0, 5.0, 5.0, 5.0]

@pytest.fixture
def roi_mask_list(image_dims, motion_border):

    base_pixels = np.argwhere(np.ones((10, 10)))

    masks = []
    for ii in range(10):
        pixels = base_pixels + ii * 10
        masks.append(roi_masks.create_roi_mask(
            image_dims['width'], 
            image_dims['height'],
            motion_border,
            pix_list=pixels,
            label=str(ii),
            mask_group=-1
        ))

    return masks

@pytest.fixture
def neuropil_masks(roi_mask_list, motion_border):
    neuropil_masks = []

    mask_array = roi_masks.create_roi_mask_array(roi_mask_list)
    combined_mask = mask_array.max(axis=0)

    for roi_mask in roi_mask_list:
        neuropil_masks.append(roi_masks.create_neuropil_mask(
            roi_mask, 
            motion_border, 
            combined_mask, 
            roi_mask.label
        ))
    return neuropil_masks

@pytest.fixture
def video(image_dims):
    num_frames = 20
    data = np.ones((num_frames, image_dims['height'], image_dims['width']))
    data[:, 50:, 50:] = 2
    return data


def test_calculate_traces(video, roi_mask_list):
    roi_traces, exclusions = roi_masks.calculate_traces(video, roi_mask_list)

    expected_exclusions = pd.DataFrame({
        'roi_id': ['0', '9'],
        'exclusion_label_name': ['motion_border', 'motion_border']
    })

    assert np.all(np.isnan(roi_traces[0, :]))
    assert np.all(roi_traces[4, :] == 1)
    assert np.all(roi_traces[6, :] == 2)
    assert np.all(np.isnan(roi_traces[9, :]))
    
    pd.testing.assert_frame_equal(expected_exclusions, pd.DataFrame(exclusions), check_like=True)


def test_validate_masks(roi_mask_list, neuropil_masks):
    roi_mask_list.extend(neuropil_masks)
    roi_mask_list[3].mask = np.zeros_like(roi_mask_list[3].mask)
    roi_mask_list[17].mask = np.zeros_like(roi_mask_list[17].mask)

    obtained = []
    for mask in roi_mask_list:
        obtained.extend(roi_masks.validate_mask(mask))

    expected_exclusions = pd.DataFrame({
        'roi_id': ['0', '3', '9', '7'],
        'exclusion_label_name': ['motion_border', 'empty_roi_mask', 'motion_border', 'empty_neuropil_mask']
    })
    pd.testing.assert_frame_equal(expected_exclusions, pd.DataFrame(obtained), check_like=True)

