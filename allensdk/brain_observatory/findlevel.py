# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def findlevel(inwave, threshold, direction='both'):
    temp = inwave - threshold
    if (direction.find("up") + 1):
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0) > 0)
    elif (direction.find("down") + 1):
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0) < 0)
    else:
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0))

    if len(crossings) == 0 or len(crossings[0]) == 0:
        return None

    return crossings[0][0]
