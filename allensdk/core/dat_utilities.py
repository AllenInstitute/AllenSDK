# Copyright 2015 Allen Institute for Brain Science
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

import numpy


class DatUtilities(object):

    @classmethod
    def save_voltage(cls, output_path, v, t):
        '''Save a single voltage output result into a simple text format.

        The output file is one t v pair per line.

        Parameters
        ----------
        output_path : string
            file name for output
        v : numpy array
            voltage
        t : numpy array
            time
        '''
        data = numpy.transpose(numpy.vstack((t, v)))
        with open(output_path, "w") as f:
            numpy.savetxt(f, data)
