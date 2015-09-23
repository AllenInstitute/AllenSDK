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

import os, unittest

from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.core.cell_types_cache import CellTypesCache

class NwbDataSetTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NwbDataSetTest, self).__init__(*args, **kwargs)

    def testAllDataSets(self):

        manifest_file = '/local1/projects/FHL2015/cell_types/manifest.json'
        if not os.path.exists(manifest_file):
            print "Cannot run this test: manifest does not exist (%s)" % manifest_file
            return True
        
        self.cache = CellTypesCache(manifest_file=manifest_file)
        cells = self.cache.get_cells()

        for cell in cells:
            data_set = self.cache.get_ephys_data(cell['id'])
            sweeps = self.cache.get_ephys_sweeps(cell['id'])

            for sweep in sweeps:
                metadata = data_set.get_sweep_metadata(sweep['sweep_number'])

if __name__ == "__main__":
    unittest.main()



