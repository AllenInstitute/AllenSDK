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

import json, sys, traceback
from allensdk.brain_observatory.brain_observatory_analysis import run_brain_observatory_analysis

def read_json(json_filename):

    try:    
        with open(json_filename, 'r') as f:
            input_data = json.load(f)
    
            for _, session in input_data.items():
                session_name = session['session_name']
                depth = session['depth']
                nwb_file = session['nwb_file']
                output_file = session['output_file']
            
                args = [session_name,
                        nwb_file,
                        output_file,
                        { 'experiment_id': None,
                          'area': None,
                          'depth': depth }]
                        
                run_brain_observatory_analysis(*args)
    except Exception as error:
        print traceback.print_exc(file=sys.stderr)

    
if __name__ == '__main__':
    json_filename = sys.argv[-1]
    
    read_json(json_filename)