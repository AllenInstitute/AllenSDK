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

