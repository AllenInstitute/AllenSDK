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

import itertools as it

import pytest
import mock

from scipy.stats import multivariate_normal
from skimage.transform import rotate
import skimage
import numpy as np

import allensdk.brain_observatory.receptive_field_analysis.fitgaussian2D as gauss


@pytest.fixture(scope='function')
def gaussian_pdf():

    def gpdf(mean, cov, axes, scale=1):

        rv = multivariate_normal( mean, cov )
        mesh = np.meshgrid( *axes, indexing='ij' )
        pos = np.rollaxis( np.array(mesh), 0, len( axes ) + 1 )

        out = rv.pdf( pos )
        out = out / np.amax(out) * scale

        return out, mesh

    return gpdf


@pytest.fixture(scope='function')
def domain_axes():

    start = 0
    stop = 201
    step = 1
    naxes = 2

    axes = [ np.arange(start, stop, step) for ii in range(naxes) ]    
    return axes


@pytest.fixture(scope='function')
def simple_fill():

    def do_fill(domain_axes, fn):

        arr = np.zeros([ len(da) for da in domain_axes ])
        for pt in it.product(*domain_axes):
            arr[pt] = fn(*pt)

        return arr

    return do_fill


@pytest.mark.parametrize('mean,cov,scale', [ [ [ 100, 100 ], [ 25, 25 ], 1 ],
                                                 [ [ 100, 100 ], [ 10, 25 ], 1 ],
                                                 [ [ 100, 110 ], [ 25, 25 ], 1 ],
                                                 [ [ 100, 110 ], [ 10, 25 ], 1 ],
                                                 [ [ 110, 100 ], [ 10, 25 ], 1 ] ])
def test_gaussian2D_norot(mean, cov, scale, gaussian_pdf, domain_axes, simple_fill):

    full_cov = [ [ cov[0], 0 ], [ 0, cov[1] ] ]
    exp, mesh = gaussian_pdf( mean, full_cov, domain_axes, scale )
    
    obt_fn = gauss.gaussian2D( scale, mean[0], mean[1], np.sqrt(cov[0]), np.sqrt(cov[1]), 0 )
    obt = simple_fill( domain_axes, obt_fn )

    assert( np.allclose( obt, exp ) )


# only providing independent cov - using rotation after the fact
@pytest.mark.skipif(skimage.__version__ < '0.11.1', reason='cannot rotate about non-center point before .11.1')
@pytest.mark.parametrize('mean,cov,scale,rot', [ [ [ 100, 100 ], [ 25, 25 ], 1, 0 ],
                                                 [ [ 100, 100 ], [ 10, 25 ], 1, 0 ],
                                                 [ [ 100, 110 ], [ 25, 25 ], 1, 0 ],
                                                 [ [ 100, 110 ], [ 10, 25 ], 1, 0 ],
                                                 [ [ 110, 100 ], [ 10, 25 ], 1, 0 ],
                                                 [ [ 100, 100 ], [ 25, 25 ], 1, 90 ],
                                                 [ [ 100, 100 ], [ 25, 20 ], 1, 180 ],
                                                 [ [ 100, 100 ], [ 30, 25 ], 1, -90 ],
                                                 [ [ 100, 110 ], [ 20, 15 ], 1, -45 ],
                                                 [ [ 100, 110 ], [ 20, 15 ], 1, 30 ], 
                                                 [ [ 100, 100 ], [ 15, 20 ], 1, 10 ],
                                                 [ [ 100, 100 ], [ 10, 25 ], 10, 0 ] ])
def test_gaussian2D(mean, cov, scale, rot, gaussian_pdf, domain_axes, simple_fill):

    full_cov = [ [ cov[0], 0 ], [ 0, cov[1] ] ]
    exp, mesh = gaussian_pdf( mean, full_cov, domain_axes, scale )

    if rot != 0:
        exp = rotate( exp, -rot, False, center=mean[::-1] ) # negative rotation
    
    obt_fn = gauss.gaussian2D( scale, mean[0], mean[1], np.sqrt(cov[0]), np.sqrt(cov[1]), rot )
    obt = simple_fill( domain_axes, obt_fn )

    if rot == 0:
        assert( np.allclose( obt, exp ) )
    else:
        assert( np.linalg.norm( obt - exp ) / np.linalg.norm(exp) < 10 ** -2 )


@pytest.mark.parametrize('mean,cov,scale', [ [ [ 100, 100 ], [ [1, 0 ], [0, 1] ], 1 ],
                                             [ [ 100, 150 ], [ [1, 0 ], [0, 1] ], 1 ],
                                             [ [ 125, 125 ], [ [1, 0 ], [0, 1] ], 1 ],
                                             [ [ 110, 100 ], [ [1, 0 ], [0, 1] ], 1 ],
                                             [ [ 90, 100 ], [ [1, 0 ], [0, 1] ], 1 ],
                                             [ [ 100, 100 ], [ [1, 0 ], [0, 1] ], 2 ],
                                             [ [ 100, 100 ], [ [5, 0 ], [0, 1] ], 1 ] ])
def test_moments2(mean, cov, scale, gaussian_pdf, domain_axes):

    pdf, mesh = gaussian_pdf( mean, cov, domain_axes, scale )
    mom_exp = np.array([ scale, 
                         mean[0], mean[1], 
                         np.sqrt(cov[1][1]), np.sqrt(cov[0][0]) ])

    mom_obt = gauss.moments2( pdf )

    assert( np.allclose( mom_obt[:-1], mom_exp ) )
    assert( mom_obt[-1] is None ) # TODO: why?
    

# we probably want to test rotation here at some point, but there is no way that it could work now, given
# that moments2 assumes independence ...
@pytest.mark.parametrize('mean,cov,scale', [ [ [ 100, 100 ], [ 25, 25 ], 1 ],
                                             [ [ 100, 100 ], [ 10, 25 ], 1 ],
                                             [ [ 100, 110 ], [ 25, 25 ], 1 ],
                                             [ [ 100, 110 ], [ 10, 25 ], 1 ],
                                             [ [ 110, 100 ], [ 10, 25 ], 1 ] ])
def test_fitgaussian2D(mean, cov, scale, gaussian_pdf, domain_axes):

    full_cov = [ [ cov[0], 0 ], [ 0, cov[1] ] ]
    img, mesh = gaussian_pdf( mean, full_cov, domain_axes, scale )

    obt = gauss.fitgaussian2D( img )
    exp = [ scale, mean[0], mean[1], np.sqrt(cov[0]), np.sqrt(cov[1]), 0 ]

    assert( np.allclose( exp, obt, atol=10**-3 ) )


def test_fitgaussian2D_failure():

    data = np.eye(10)

    res = mock.MagicMock()
    res.success = False
    res.status = 3
    res.message = 'foo'

    with mock.patch('scipy.optimize.minimize', return_value=res) as p:
        with pytest.raises( gauss.GaussianFitError ):
            gauss.fitgaussian2D(data)
