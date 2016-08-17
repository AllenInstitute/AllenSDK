# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import allensdk.brain_observatory.dff as dff
import numpy as np


def test_movingmode_fast():
    # check basic behavior
    x = np.array([0, 10, 0, 0, 20, 0, 0, 0, 30])
    kernelsize = 4
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(y == 0)

    # check window edges
    x = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    kernelsize = 2
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(x == y)

    # check > 16 bit
    x = np.array([4097, 4097, 4097, 4097])
    kernelsize = 2
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(y == 4097)

    # check floats
    x = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.float32)
    kernelsize = 2
    y = np.zeros(x.shape)

    dff.movingmode_fast(x, kernelsize, y)

    assert np.all(x == y)
