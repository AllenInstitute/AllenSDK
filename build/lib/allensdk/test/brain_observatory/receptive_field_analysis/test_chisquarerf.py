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

import os

import pytest

from scipy.ndimage.interpolation import zoom
import scipy.stats as stats
import numpy as np

from allensdk.brain_observatory.receptive_field_analysis import chisquarerf as chi


@pytest.fixture
def rf_events():
    
    np.random.seed(12)

    def make(receptive_field_mask, lsn):
        activity = np.logical_or(lsn == 255, lsn==0)
        return np.logical_and(activity, receptive_field_mask).sum(axis=(1, 2))[:, None]

    return make


@pytest.fixture
def locally_sparse_noise():

    def make(ntr, nr, nc):
        return np.around(np.random.rand(ntr, nr, nc)*255).astype(int)

    return make


@pytest.fixture
def rf_mask():

    def make(nr, nc, slices):
        mask = np.zeros((nr, nc))
        mask[slices] = 1
        return mask
  
    return make


@pytest.fixture
def exclusion_mask():
    mask = np.zeros((4, 4, 2))
    mask[:, :2, :] = 1
    return mask


@pytest.fixture
def events_per_pixel():

    epp = np.zeros((2, 4, 4, 2))
    epp[0, 0, 0, 0] = 2
    epp[0, 0, 0, 1] = 3
    epp[1, 3, 3, 0] = 5
    epp[1, 1, 0, 0] = 4
  
    return epp


@pytest.fixture
def trials_per_pixel():

    tpp = np.zeros((4, 4, 2))
    tpp[:, :, 0] = 2
    tpp[:, :, 1] = 0

    return tpp


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


def test_build_trial_matrix():

    tr0 = np.eye(16) * 255
    tr1 = np.arange(256).reshape((16, 16))
    lsn_template = np.array([ tr0, tr1 ])

    exp = np.zeros((16, 16, 2, 2))
    exp[:, :, 0, 0] = np.eye(16)
    exp[:, :, 1, 0] = 1 - np.eye(16)
    exp[15, 15, 0, 1] = 1
    exp[0, 0, 1, 1] = 1

    obt = chi.build_trial_matrix( lsn_template, 2 )
    assert(np.allclose( exp, obt ))


def test_get_expected_events_by_pixel(exclusion_mask, events_per_pixel, trials_per_pixel):

    obt = chi.get_expected_events_by_pixel(exclusion_mask, events_per_pixel, trials_per_pixel)

    assert( obt[0, 0, 0, 0] == 0.625 ) # 5 events, 8 trials (events counted even if 0 trials)
    assert( obt[1, 1, 0, 0] == 0.5 ) # 4 events, 8 trials
    assert( obt[0, 0, 0, 1] == 0.0 ) # no trials
    assert( obt[1, 3, 3, 0] == 0.0 ) # out of mask


def test_chi_square_within_mask(exclusion_mask, events_per_pixel, trials_per_pixel):

    obt_p, obt_ch = chi.chi_square_within_mask(exclusion_mask, events_per_pixel, trials_per_pixel)

    resps = np.array([4, 0, 0, 0, 0, 0, 0, 0])
    resids = resps - 0.5
    chi_sum = (resids ** 2 / 0.5).sum()

    exp_p = 1.0 - stats.chi2.cdf(chi_sum, 15)

    # the zeroth test cell has a response without a trial.
    # this is infinitely surprising, so the pval is 0
    assert(np.allclose( obt_p, [0, exp_p] )) 


def test_get_disc_masks():

    lsn_template = np.zeros((9, 3, 3)) + 128
    for ii in range(3):
        for jj in range(3):
            lsn_template[3*ii+jj, ii, jj] = 0
    lsn_template[4, 2, 2] = 255

    exp1 = np.ones((3, 3))
    exp1[2, 2] = 0

    exp0 = np.zeros((3, 3))
    exp0[:2, :2] = 1

    obt = chi.get_disc_masks(lsn_template, radius=1)

    assert(np.allclose( exp1, obt[1, 1, :, :] ))
    assert(np.allclose( exp0, obt[0, 0, :, :] ))


def test_get_events_per_pixel():
    
    events = np.zeros((3, 2))
    trials = np.zeros((4, 4, 2, 3))

    # pixel 1,1 is off trial 1 and on trial 2 
    trials[1, 1, 1, 1] = 1
    trials[1, 1, 0, 2] = 1

    # pixel 2,2 is on trial 2 and off trial 0
    trials[2, 2, 0, 2] = 1
    trials[2, 2, 1, 0] = 1

    # cell 0 has 4 events on trial 2 and 1 on trial 0
    events[2, 0] = 4
    events[0, 0] = 1
  
    # cell 1 has 2 events on trial 1
    events[1, 1] = 2

    exp = np.zeros((2, 4, 4, 2))
    exp[0, 2, 2, 0] = 4
    exp[0, 1, 1, 0] = 4
    exp[0, 2, 2, 1] = 1
    exp[1, 1, 1, 1] = 2

    obt = chi.get_events_per_pixel(events, trials)
    assert(np.allclose( obt, exp ))


@pytest.mark.parametrize('base,ex', [[5., 10], [0.1, 12], [np.arange(20), np.linspace(0, 1, 20)]])
def test_nll_to_pvalue(base, ex):

    obt = chi.NLL_to_pvalue(ex, base)
    exp = np.power(base, -ex)

    assert(np.allclose( exp, obt ))


# test by reversing nll_to_pvalue
@pytest.mark.parametrize('base,ex', [[10., 2], [10., 4], [np.array([10, 10, 10]), np.linspace(0, 1, 3)]])
def test_pvalue_to_nll(base, ex):

    pv = chi.NLL_to_pvalue(ex, base)
    max_nll = np.amax(ex)

    obt = chi.pvalue_to_NLL(pv, max_nll)

    assert(np.allclose( ex, obt ))


@pytest.mark.skipif(os.getenv('NO_TEST_RANDOM') == 'true', reason="random seed may not produce the same results on all machines")
def test_chi_square_binary(locally_sparse_noise, rf_events, rf_mask):

    ntr = 2000
    nr = 20
    nc = 20
    slices = [slice(9, 11), slice(9, 11)]

    mask = rf_mask(nr, nc, slices)
    lsn = locally_sparse_noise(ntr, nr, nc)
    events = rf_events(mask, lsn)

    obt = chi.chi_square_binary(events, lsn)
    assert( obt[0][slices].sum() == 0 )
    assert( obt.sum() > 0 )


@pytest.mark.skipif(os.getenv('NO_TEST_RANDOM') == 'true', reason="random seed may not produce the same results on all machines")
def test_get_peak_significance(locally_sparse_noise, rf_events, rf_mask):

    ntr = 2000
    nr = 20
    nc = 20
    slices = [slice(9, 11), slice(9, 11)]

    mask = rf_mask(nr, nc, slices)
    lsn = locally_sparse_noise(ntr, nr, nc)
    events = rf_events(mask, lsn)

    chi_pv = chi.chi_square_binary(events, lsn)
    chi_nll = chi.pvalue_to_NLL(chi_pv)

    significant_cells, best_p, _, _ = chi.get_peak_significance(chi_nll, lsn)

    assert(np.allclose( best_p, 0 ))
    assert(np.allclose( significant_cells, [True] ))


def test_locate_median():

    mask = np.eye(9)
    where = np.where(mask)
    
    obt = chi.locate_median(*where)
    assert(np.allclose( obt , [4, 4] ))
