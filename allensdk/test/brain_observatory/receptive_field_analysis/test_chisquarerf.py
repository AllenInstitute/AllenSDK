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

import pytest

from scipy.ndimage.interpolation import zoom
import numpy as np

from allensdk.brain_observatory.receptive_field_analysis import chisquarerf as chi


# not testing d < 1 here
@pytest.mark.parametrize('r,c,d', [[2, 3, 4], [28, 16, 3], [28, 16, 2], [10, 20, 12]])
def test_interpolate_rf(r, c, d):

    image = np.arange( r * c ).reshape([ r, c ])

    delta_col = 1.0 / d
    delta_row = c * delta_col

    obtained = chi.interpolate_RF(image, d)
    grad = np.gradient(obtained)

    assert(np.allclose( grad[0], np.zeros_like(grad[0]) + delta_row ))
    assert(np.allclose( grad[1], np.zeros_like(grad[1]) + delta_col ))


# tests integration with interpolate
# not testing case where r, c are small
@pytest.mark.parametrize('r,c,d', [[28, 16, 3], [28, 16, 2], [10, 20, 12]])
def test_deinterpolate_rf(r, c, d):

    image = np.arange( r * c ).reshape([ r, c ])

    interp = chi.interpolate_RF(image, d)
    obt = chi.deinterpolate_RF(interp, c, r, d)

    assert(np.allclose( image, obt ))


def test_smooth_sta():

    image = np.eye(10)

    smoothed = chi.smooth_STA(image)
    
    thresholded = smoothed.copy()
    thresholded[thresholded < 0.5] = 0
    thresholded[thresholded > 0.5] = 1

    assert(np.allclose( smoothed.T, smoothed ))
    assert(np.allclose( image, thresholded ))
    assert( np.count_nonzero(smoothed) > np.count_nonzero(image) )


#def test_build_trial_matrix():
