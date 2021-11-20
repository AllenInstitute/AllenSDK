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
import allensdk.brain_observatory.circle_plots as cplots
import numpy as np

def test_polar_to_xy():
    d = cplots.polar_to_xy([0], 0.0)
    assert d.shape[0] == 1
    assert d.shape[1] == 2

    d = cplots.polar_to_xy([0, np.pi], 1.0)

    assert np.allclose(d, [[ 1, 0 ], [-1, 0]])

def test_polar_linspace():
    d = cplots.polar_linspace(1, 0, 180, 2, endpoint=True, degrees=True)
    assert np.allclose(d, [[1,0],[-1,0]])

    d = cplots.polar_linspace(1, 0, np.pi, 2, endpoint=False, degrees=False)
    assert np.allclose(d, [[1,0],[0,1.0]])

    d = cplots.polar_linspace(2, 0, 2*np.pi, 4, endpoint=False, degrees=False)
    assert np.allclose(d, [[2,0],[0,2],[-2,0],[0,-2]])

    d = cplots.polar_linspace(3, 0, 360, 5, endpoint=True, degrees=True)
    assert np.allclose(d, [[3,0],[0,3],[-3,0],[0,-3],[3,0]])

def test_spiral_trials():
    coll = cplots.spiral_trials([0,2])

    assert len(coll.get_paths()) == 2

def test_spiral_trials_polar():
    coll = cplots.spiral_trials_polar(1.0, 0.0, [1.0])
    assert len(coll.get_paths()) == 1

    coll = cplots.spiral_trials_polar(1.0, 0.0, [1.0], offset=[1,0])
    assert len(coll.get_paths()) == 1

def test_angle_lines():
    lines = cplots.angle_lines([0], 0, 1)
    assert len(lines.get_paths()) == 1

    lines = cplots.angle_lines([0,1], 0, 1)
    assert len(lines.get_paths()) == 2

def test_radial_arcs():
    arcs = cplots.radial_arcs([1], 0, 1)
    assert len(arcs.get_paths()) == 1

    arcs = cplots.radial_arcs([1,2], 0, 1)
    assert len(arcs.get_paths()) == 2

def test_radial_circles():
    d = cplots.radial_circles([1])
    assert len(d.get_paths()) == 1

    d = cplots.radial_circles([1,2])
    assert len(d.get_paths()) == 2

def test_polar_line_circles():
    d = cplots.polar_line_circles([1],0)
    assert len(d.get_paths()) == 1

    d = cplots.polar_line_circles([1],0,0)
    assert len(d.get_paths()) == 1

def test_wedge_ring():
    d = cplots.wedge_ring(1, 0, 1, 0, 180)
    assert len(d.get_paths()) == 1

    d = cplots.wedge_ring(2, 0, 1)
    assert len(d.get_paths()) == 2

def test_reset_hex_pack():
    pos = cplots.hex_pack(1.0, 1)
    cplots.reset_hex_pack()
    assert len(cplots.HEX_POSITIONS) == 0
    
def test_hex_pack():
    cplots.reset_hex_pack()

    pos = cplots.hex_pack(1.0, 1)
    assert pos.shape[0] == 1
    assert np.allclose(pos, [[0,0]])
    assert np.allclose(cplots.HEX_POSITIONS.shape, [1,2])

    pos = cplots.hex_pack(2.0, 2)
    assert np.allclose(pos, [[0,0],[4,0]])
    assert np.allclose(cplots.HEX_POSITIONS.shape, [7,2])

    pos = cplots.hex_pack(2.0, 8)
    assert np.allclose(cplots.HEX_POSITIONS.shape, [19,2])




