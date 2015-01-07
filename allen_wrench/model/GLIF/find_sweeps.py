import json, sys
import argparse

SHORT_SQUARE = 'Short Square'
RAMP = 'Ramp'
NOISE1 = 'Noise1'
NOISE2 = 'Noise2'
MULTI_SHORT_SQUARE = 'Multi Short Square'
RAMP_TO_RHEO = 'Ramp To Rheo'

RETURN_CODE = 0

def soft_fail(msg):
    global RETURN_CODE
    RETURN_CODE = 1
    print "WARNING:", msg

def get_sweep_numbers(sweep_list):
    return [ s['sweep_number'] for s in sweep_list]

def get_sweeps_by_type(sweep_list, sweep_type):
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

def find_short_square_sweeps(sweep_list):
    '''
    Find 1) all of the subthreshold short square sweeps
         2) all of the superthreshold short square sweeps
         3) the subthresholds short square sweep with maximum stimulus amplitude
    '''
    short_square_sweeps = get_sweeps_by_type(sweep_list, SHORT_SQUARE)
    subthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) > 0 ]
    multi_short_square_sweeps = get_sweeps_by_type(sweep_list, MULTI_SHORT_SQUARE)
    
    out = {
        'all_short_square': get_sweep_numbers(short_square_sweeps),
        'subthreshold_short_square': get_sweep_numbers(subthreshold_short_square_sweeps),
        'superthreshold_short_square': get_sweep_numbers(superthreshold_short_square_sweeps),
        'multi_short_square': get_sweep_numbers(multi_short_square_sweeps),
        'maximum_subthreshold_short_square': find_ranked_sweep(subthreshold_short_square_sweeps, 'stimulus_amplitude', reverse=True),
        'minimum_superthreshold_short_square': find_ranked_sweep(superthreshold_short_square_sweeps, 'stimulus_amplitude')

    }

    if len(out['maximum_subthreshold_short_square']) == 0: 
        soft_fail("no passed maximum subthreshold short square")

    if len(out['minimum_superthreshold_short_square']) == 0:
        soft_fail("no passed minimum short square")

    return out

def find_ramp_sweeps(sweep_list):
    '''
    Find 1) all ramp sweeps
         2) all subthreshold ramps
         3) all superthreshold ramps
    '''
    ramp_sweeps = get_sweeps_by_type(sweep_list, RAMP)
    subthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) > 0 ]
    ramp_to_rheo_sweeps = get_sweeps_by_type(sweep_list, RAMP_TO_RHEO)

    out = { 
        'all_ramps': get_sweep_numbers(ramp_sweeps),
        'subthreshold_ramp': get_sweep_numbers(subthreshold_ramp_sweeps),
        'superthreshold_ramp': get_sweep_numbers(superthreshold_ramp_sweeps),
        'ramp_to_rheo': get_sweep_numbers(ramp_to_rheo_sweeps),
        'maximum_subthreshold_ramp': find_ranked_sweep(subthreshold_ramp_sweeps, 'stimulus_amplitude', reverse=True)
    }

    if len(out['superthreshold_ramp']) == 0:
        soft_fail("no passing superthreshold ramp")

    return out
    
def find_noise_sweeps(sweep_list):
    '''
    Find 1) the 1st noise1 sweep (run 1)
         2) the 2nd noise1 sweep (run 2)
         2) the 1st noise2 sweep (run 1)
         3) the 2nd noise2 sweep (run 2)
         4) all noise sweeps
    '''

    noise1_sweeps = get_sweeps_by_type(sweep_list, NOISE1)
    noise2_sweeps = get_sweeps_by_type(sweep_list, NOISE2)

    all_noise_sweeps = sorted(noise1_sweeps + noise2_sweeps, key=lambda x: x['sweep_number'])

    out = {
        'all_noise': get_sweep_numbers(all_noise_sweeps)
    }

    num_noise1_sweeps = len(noise1_sweeps)
    num_noise2_sweeps = len(noise2_sweeps)

    if num_noise1_sweeps >= 2:
        noise1_sweep_numbers = get_sweep_numbers(noise1_sweeps)
        out['noise1_run1'] = [ noise1_sweep_numbers[0] ]
        out['noise1_run2'] = [ noise1_sweep_numbers[1] ]
    else:
        soft_fail("not enough noise1 sweeps (%d)" % (num_noise1_sweeps))

    if num_noise2_sweeps >= 2:
        noise2_sweep_numbers = get_sweep_numbers(noise2_sweeps)
        out['noise2_run1'] = [ noise2_sweep_numbers[0] ]
        out['noise2_run2'] = [ noise2_sweep_numbers[1] ]
    else:
        soft_fail("not enough noise2 sweeps (%d)" % (num_noise2_sweeps))
        
    return out

def filter_sweep_list(sweep_list):
    sorted_sweeps = sorted(sweep_list, key=lambda x: x['sweep_number'] ) 
    output_sweeps = [ { 
        'resting_potential': sweep.get('slow_vm_mv', None),
        'num_spikes': sweep.get('num_spikes', None),
        'stimulus_type': sweep.get('stimulus_type', None),
        'stimulus_amplitude': sweep.get('stimulus_amplitude', None),
        'sweep_number': sweep['sweep_number'],
        'workflow_state': sweep.get('workflow_state',None)
    } for sweep in sorted_sweeps ]

    return output_sweeps

def find_failed_sweeps(sweep_list, data):
    out_data = {}

    for k,v in data.iteritems():
        if all([ isinstance(vi, int) for vi in v ]):
            out_data[k] = [ sweep_list[vi].get('workflow_state',None) for vi in v ]

    return out_data

def main():
    parser = argparse.ArgumentParser(description='find relevant sweeps from a sweep catalog')

    parser.add_argument('sweep_file', help='json file containing a list of sweeps for a cell')
    parser.add_argument('output_file', help='output json data config file')

    args = parser.parse_args()

    try:
        assert args.sweep_file is not None, Exception("A sweep configuration file name required.")
        assert args.output_file is not None, Exception("An output file name is required.")
    except Exception, e:
        parser.print_help()
        exit(1)

    input_data = None
    with open(args.sweep_file, 'rb') as f:
        input_data = json.loads(f.read())

    data = {
        'filename': input_data['filename'],
        'sweeps': filter_sweep_list(input_data['sweeps'])
    }
    
    data.update(find_short_square_sweeps(data['sweeps']))
    data.update(find_ramp_sweeps(data['sweeps']))
    data.update(find_noise_sweeps(data['sweeps']))

    data['failed_sweeps'] = find_failed_sweeps(data['sweeps'], data)

    with open(args.output_file, 'wb') as f:
        f.write(json.dumps(data, indent=2))

    sys.exit(RETURN_CODE)

if __name__ == "__main__":  main()
