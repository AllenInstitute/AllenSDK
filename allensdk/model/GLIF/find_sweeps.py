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

import json, sys, os
import logging
import argparse
import utilities

from utilities import MissingSweepException

SHORT_SQUARE = 'Short Square'
SHORT_SQUARE_60 = 'Short Square - Hold -60mv'
SHORT_SQUARE_80 = 'Short Square - Hold -80mv'
RAMP = 'Ramp'
NOISE1 = 'Noise 1'
NOISE2 = 'Noise 2'
SHORT_SQUARE_TRIPLE = 'Short Square - Triple'
RAMP_TO_RHEO = 'Ramp to Rheobase'

FILE_TYPE = 'ORCA'
PASSED_WORKFLOW_STATE = 'auto_passed'
SUCCESS=0

def fail_missing_sweep(msg, validate):
    logging.error(msg)
    global SUCCESS
    SUCCESS=1
    
    if validate:
        raise MissingSweepException(msg)

def get_sweep_numbers(sweep_list):
    return [ s['sweep_number'] for s in sweep_list]

def get_sweep_stimulus_type(sweep):
    try:
        return sweep['ephys_stimulus']['ephys_stimulus_type']['name']
    except:
        return None

def get_sweeps_by_type(sweeps, sweep_type):
    return [ s for sn, s in sweeps.iteritems() if get_sweep_stimulus_type(s) == sweep_type ]

def find_ranked_sweep(sweep_list, key, reverse=False):
    if sweep_list:
        sorted_sweep_list = sorted(sweep_list, key=lambda x: x[key], reverse=reverse)
        
        out_sweeps = [ sorted_sweep_list[0] ]
 
        for i in xrange(1,len(sweep_list)):
            if sorted_sweep_list[i][key] == out_sweeps[0][key]:
                out_sweeps.append(sorted_sweep_list[i])
            else:
                break
        
        return get_sweep_numbers(out_sweeps)
    else:
        return []

def find_short_square_sweeps(sweeps, validate):
    '''
    Find 1) all of the subthreshold short square sweeps
         2) all of the superthreshold short square sweeps
         3) the subthresholds short square sweep with maximum stimulus amplitude
    '''
    short_square_sweeps = get_sweeps_by_type(sweeps, SHORT_SQUARE)
    subthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) > 0 ]
    short_square_triple_sweeps = get_sweeps_by_type(sweeps, SHORT_SQUARE_TRIPLE)

    short_square_60_sweeps = get_sweeps_by_type(sweeps, SHORT_SQUARE_60)
    short_square_80_sweeps = get_sweeps_by_type(sweeps, SHORT_SQUARE_80)
    
    out = {
        'all_short_square': get_sweep_numbers(short_square_sweeps),
        'subthreshold_short_square': get_sweep_numbers(subthreshold_short_square_sweeps),
        'superthreshold_short_square': get_sweep_numbers(superthreshold_short_square_sweeps),
        'short_square_triple': get_sweep_numbers(short_square_triple_sweeps),
        'short_square_60': get_sweep_numbers(short_square_60_sweeps),
        'short_square_80': get_sweep_numbers(short_square_80_sweeps),
        'maximum_subthreshold_short_square': find_ranked_sweep(subthreshold_short_square_sweeps, 'stimulus_amplitude', reverse=True),
        'minimum_superthreshold_short_square': find_ranked_sweep(superthreshold_short_square_sweeps, 'stimulus_amplitude')

    }

    if len(out['maximum_subthreshold_short_square']) == 0: 
        fail_missing_sweep("No passed maximum subthreshold short square", validate)

    if len(out['minimum_superthreshold_short_square']) == 0:
        fail_missing_sweep("No passed minimum superthreshold short square", validate)

    return out

def find_ramp_sweeps(sweeps, validate):
    '''
    Find 1) all ramp sweeps
         2) all subthreshold ramps
         3) all superthreshold ramps
    '''
    ramp_sweeps = get_sweeps_by_type(sweeps, RAMP)
    subthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) > 0 ]
    ramp_to_rheo_sweeps = get_sweeps_by_type(sweeps, RAMP_TO_RHEO)

    out = { 
        'all_ramps': get_sweep_numbers(ramp_sweeps),
        'subthreshold_ramp': get_sweep_numbers(subthreshold_ramp_sweeps),
        'superthreshold_ramp': get_sweep_numbers(superthreshold_ramp_sweeps),
        'ramp_to_rheo': get_sweep_numbers(ramp_to_rheo_sweeps),
        'maximum_subthreshold_ramp': find_ranked_sweep(subthreshold_ramp_sweeps, 'stimulus_amplitude', reverse=True)
    }

    if len(out['superthreshold_ramp']) == 0:
        fail_missing_sweep("no passing superthreshold ramp", validate)

    return out
    
def find_noise_sweeps(sweeps, validate):
    '''
    Find 1) the noise1 sweeps
         2) the noise2 sweeps
         4) all noise sweeps
    '''

    noise1_sweeps = get_sweeps_by_type(sweeps, NOISE1)
    noise2_sweeps = get_sweeps_by_type(sweeps, NOISE2)

    all_noise_sweeps = sorted(noise1_sweeps + noise2_sweeps, key=lambda x: x['sweep_number'])

    out = {
        'noise1': get_sweep_numbers(noise1_sweeps),
        'noise2': get_sweep_numbers(noise2_sweeps),
        'all_noise': get_sweep_numbers(all_noise_sweeps)
    }

    num_noise1_sweeps = len(noise1_sweeps)
    num_noise2_sweeps = len(noise2_sweeps)

    if num_noise1_sweeps < 3:
        fail_missing_sweep("not enough noise1 sweeps (%d)" % (num_noise1_sweeps), validate)

    if num_noise2_sweeps < 3:
        fail_missing_sweep("not enough noise2 sweeps (%d)" % (num_noise2_sweeps), validate)
        
    return out


def find_sweeps(data_file_name, sweeps, validate):
    
    if validate:
        sweeps = [ s for s in sweeps if s['workflow_state'] == PASSED_WORKFLOW_STATE ]

    data = {
        'filename': data_file_name,
        'sweeps': { s['sweep_number']: s for s in sweeps }
    }

    data.update(find_short_square_sweeps(data['sweeps'], validate))
    data.update(find_ramp_sweeps(data['sweeps'], validate))
    data.update(find_noise_sweeps(data['sweeps'], validate))

    return data
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='find relevant sweeps from a sweep catalog')

    parser.add_argument('sweep_file', help='json file containing a list of sweeps for a cell')
    parser.add_argument('output_file', help='output json data config file')
    parser.add_argument('--no_validate', help="don't throw an exception if there was a problem", action='store_true')

    args = parser.parse_args()

    try:
        assert args.sweep_file is not None, Exception("A sweep configuration file name required.")
        assert args.output_file is not None, Exception("An output file name is required.")
    except Exception, e:
        parser.print_help()
        sys.exit(1)

    return args

def extract_input_fields(data):
    well_known_files = data['specimen']['ephys_roi_result']['well_known_files']

    file_name = None
    for wkf in well_known_files:
        if wkf['well_known_file_type']['name'] == FILE_TYPE:
            file_name = os.path.join(wkf['storage_directory'], wkf['filename'])
            break

    assert file_name, Exception("Could not find data file.")

    sweeps = data['specimen']['ephys_sweeps']

    return file_name, sweeps


def main():
    args = parse_arguments()

    input_data = None
    utilities.read_json(args.sweep_file)
    file_name, sweeps = extract_input_fields(input_data)

    data = find_sweeps(file_name, sweeps, not args.no_validate)

    utilities.write_json(args.output_file, data)

    sys.exit(SUCCESS)


if __name__ == "__main__":  main()
