import json, sys, os
import logging
import argparse

SHORT_SQUARE = 'Short Square'
SHORT_SQUARE_60 = 'Short Square - Hold -60mv'
SHORT_SQUARE_80 = 'Short Square - Hold -80mv'
RAMP = 'Ramp'
NOISE1 = 'Noise 1'
NOISE2 = 'Noise 2'
SHORT_SQUARE_TRIPLE = 'Short Square - Triple'
RAMP_TO_RHEO = 'Ramp to Rheobase'

FILE_TYPE = 'ORCA'

def fail(msg, validate):
    logging.error(msg)

    if validate:
        raise Exception(msg)

def get_sweep_numbers(sweep_list):
    return [ s['sweep_number'] for s in sweep_list]

def get_sweeps_by_type(sweep_list, sweep_type, validate):
    if validate:
        return [ s for s in sweep_list if s.get('stimulus_type',None) == sweep_type and s.get('workflow_state',None) == 'passed' ]
    else:
        return [ s for s in sweep_list if s.get('stimulus_type',None) == sweep_type ]

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

def find_short_square_sweeps(sweep_list, validate):
    '''
    Find 1) all of the subthreshold short square sweeps
         2) all of the superthreshold short square sweeps
         3) the subthresholds short square sweep with maximum stimulus amplitude
    '''
    short_square_sweeps = get_sweeps_by_type(sweep_list, SHORT_SQUARE, validate)
    subthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) > 0 ]
    short_square_triple_sweeps = get_sweeps_by_type(sweep_list, SHORT_SQUARE_TRIPLE, validate)

    short_square_60_sweeps = get_sweeps_by_type(sweep_list, SHORT_SQUARE_60, validate)
    short_square_80_sweeps = get_sweeps_by_type(sweep_list, SHORT_SQUARE_80, validate)
    
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
        fail("No passed maximum subthreshold short square", validate)

    if len(out['minimum_superthreshold_short_square']) == 0:
        fail("No passed minimum superthreshold short square", validate)

    return out

def find_ramp_sweeps(sweep_list, validate):
    '''
    Find 1) all ramp sweeps
         2) all subthreshold ramps
         3) all superthreshold ramps
    '''
    ramp_sweeps = get_sweeps_by_type(sweep_list, RAMP, validate)
    subthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) > 0 ]
    ramp_to_rheo_sweeps = get_sweeps_by_type(sweep_list, RAMP_TO_RHEO, validate)

    out = { 
        'all_ramps': get_sweep_numbers(ramp_sweeps),
        'subthreshold_ramp': get_sweep_numbers(subthreshold_ramp_sweeps),
        'superthreshold_ramp': get_sweep_numbers(superthreshold_ramp_sweeps),
        'ramp_to_rheo': get_sweep_numbers(ramp_to_rheo_sweeps),
        'maximum_subthreshold_ramp': find_ranked_sweep(subthreshold_ramp_sweeps, 'stimulus_amplitude', reverse=True)
    }

    if len(out['superthreshold_ramp']) == 0:
        fail("no passing superthreshold ramp", validate)

    return out
    
def find_noise_sweeps(sweep_list, validate):
    '''
    Find 1) the 1st noise1 sweep (run 1)
         2) the 2nd noise1 sweep (run 2)
         2) the 1st noise2 sweep (run 1)
         3) the 2nd noise2 sweep (run 2)
         4) all noise sweeps
    '''

    noise1_sweeps = get_sweeps_by_type(sweep_list, NOISE1, validate)
    noise2_sweeps = get_sweeps_by_type(sweep_list, NOISE2, validate)

    all_noise_sweeps = sorted(noise1_sweeps + noise2_sweeps, key=lambda x: x['sweep_number'])

    out = {
        'all_noise': get_sweep_numbers(all_noise_sweeps)
    }

    num_noise1_sweeps = len(noise1_sweeps)
    num_noise2_sweeps = len(noise2_sweeps)

    if num_noise1_sweeps >= 3:
        noise1_sweep_numbers = get_sweep_numbers(noise1_sweeps)
        out['noise1_run1'] = [ noise1_sweep_numbers[0] ]
        out['noise1_run2'] = [ noise1_sweep_numbers[1] ]
        out['noise1_run3'] = [ noise1_sweep_numbers[2] ]
    else:
        fail("not enough noise1 sweeps (%d)" % (num_noise1_sweeps), validate)

    if num_noise2_sweeps >= 3:
        noise2_sweep_numbers = get_sweep_numbers(noise2_sweeps)
        out['noise2_run1'] = [ noise2_sweep_numbers[0] ]
        out['noise2_run2'] = [ noise2_sweep_numbers[1] ]
        out['noise2_run3'] = [ noise2_sweep_numbers[2] ]
    else:
        fail("not enough noise2 sweeps (%d)" % (num_noise2_sweeps), validate)
        
    return out

def find_failed_sweeps(sweep_list, data):
    out_data = {}

    for k,v in data.iteritems():
        if all([ isinstance(vi, int) for vi in v ]):
            out_data[k] = [ sweep_list[vi].get('workflow_state',None) for vi in v ]

    return out_data

def find_sweeps(data_file_name, sweeps, validate):
    data = {
        'filename': data_file_name,
        'sweeps': sweeps
    }
    
    data.update(find_short_square_sweeps(data['sweeps'], validate))
    data.update(find_ramp_sweeps(data['sweeps'], validate))
    data.update(find_noise_sweeps(data['sweeps'], validate))

    data['failed_sweeps'] = find_failed_sweeps(data['sweeps'], data)

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
    with open(args.sweep_file, 'rb') as f:
        input_data = json.loads(f.read())
        file_name, sweeps = extract_input_fields(input_data)

    data = find_sweeps(file_name, sweeps, not args.no_validate)

    with open(args.output_file, 'wb') as f:
        f.write(json.dumps(data, indent=2))


if __name__ == "__main__":  main()
