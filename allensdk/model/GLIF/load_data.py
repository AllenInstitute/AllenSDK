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

from allensdk.core.nwb_data_set import NwbDataSet as EphysDataSet

def load_sweeps(file_name, sweep_numbers):
    data = [ load_sweep(file_name, sweep_number) for sweep_number in sweep_numbers ]

    return {
        'voltage': [ d['voltage'] for d in data ],
        'current': [ d['current'] for d in data ],
        'dt': [ d['dt'] for d in data ],
        'start_idx': [ d['start_idx'] for d in data ],
    }
    
def load_sweep(file_name, sweep_number):
    ds = EphysDataSet(file_name)
    data = ds.get_sweep(sweep_number)

    return {
        'current': data['stimulus'],
        'voltage': data['response'],
        'start_idx': data['index_range'][0],
        'dt': 1.0/data['sampling_rate']
    }
