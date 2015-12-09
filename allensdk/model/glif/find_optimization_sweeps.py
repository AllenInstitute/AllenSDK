import json, sys, os
import logging
import argparse
import allensdk.core.json_utilities as ju


SHORT_SQUARE = 'Short Square'
SHORT_SQUARE_60 = 'Short Square - Hold -60mv'
SHORT_SQUARE_80 = 'Short Square - Hold -80mv'
RAMP = 'Ramp'
NOISE1 = 'Noise 1'
NOISE2 = 'Noise 2'
SHORT_SQUARE_TRIPLE = 'Short Square - Triple'
RAMP_TO_RHEO = 'Ramp to Rheobase'


class MissingSweepException( Exception ): pass


def get_sweep_numbers(sweep_list):
    return [ s['sweep_number'] for s in sweep_list]


def get_sweeps_by_name(sweeps, sweep_name):
    if isinstance(sweeps, dict):
        return [ s for sn,s in sweeps.iteritems() if s[u'stimulus_name'] == sweep_name ]    
    else:
        return [ s for s in sweeps if s[u'stimulus_name'] == sweep_name ]


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


def find_short_square_sweeps(sweeps):
    '''
    Find 1) all of the subthreshold short square sweeps
         2) all of the superthreshold short square sweeps
         3) the subthresholds short square sweep with maximum stimulus amplitude
    '''
    short_square_sweeps = get_sweeps_by_name(sweeps, SHORT_SQUARE)
    subthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_short_square_sweeps = [ s for s in short_square_sweeps if s.get('num_spikes',None) > 0 ]
    short_square_triple_sweeps = get_sweeps_by_name(sweeps, SHORT_SQUARE_TRIPLE)

    short_square_60_sweeps = get_sweeps_by_name(sweeps, SHORT_SQUARE_60)
    short_square_80_sweeps = get_sweeps_by_name(sweeps, SHORT_SQUARE_80)
    
    out = {
        'all_short_square': get_sweep_numbers(short_square_sweeps),
        'subthreshold_short_square': get_sweep_numbers(subthreshold_short_square_sweeps),
        'superthreshold_short_square': get_sweep_numbers(superthreshold_short_square_sweeps),
        'short_square_triple': get_sweep_numbers(short_square_triple_sweeps),
        'short_square_60': get_sweep_numbers(short_square_60_sweeps),
        'short_square_80': get_sweep_numbers(short_square_80_sweeps),
        'maximum_subthreshold_short_square': find_ranked_sweep(subthreshold_short_square_sweeps, 'stimulus_absolute_amplitude', reverse=True),
        'minimum_superthreshold_short_square': find_ranked_sweep(superthreshold_short_square_sweeps, 'stimulus_absolute_amplitude')

    }

    errors = []
    if len(out['maximum_subthreshold_short_square']) == 0: 
        errors.append("No maximum subthreshold short square")

    if len(out['minimum_superthreshold_short_square']) == 0:
        errors.append("No minimum superthreshold short square")

    return out, errors


def find_ramp_sweeps(sweeps):
    '''
    Find 1) all ramp sweeps
         2) all subthreshold ramps
         3) all superthreshold ramps
    '''
    ramp_sweeps = get_sweeps_by_name(sweeps, RAMP)
    subthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) == 0 ]
    superthreshold_ramp_sweeps = [ s for s in ramp_sweeps if s.get('num_spikes',None) > 0 ]
    ramp_to_rheo_sweeps = get_sweeps_by_name(sweeps, RAMP_TO_RHEO)

    out = { 
        'all_ramps': get_sweep_numbers(ramp_sweeps),
        'subthreshold_ramp': get_sweep_numbers(subthreshold_ramp_sweeps),
        'superthreshold_ramp': get_sweep_numbers(superthreshold_ramp_sweeps),
        'ramp_to_rheo': get_sweep_numbers(ramp_to_rheo_sweeps),
        'maximum_subthreshold_ramp': find_ranked_sweep(subthreshold_ramp_sweeps, 'stimulus_absolute_amplitude', reverse=True)
    }

    errors = []
    if len(out['superthreshold_ramp']) == 0:
        errors.append("no superthreshold ramp")

    return out, errors
    
def find_noise_sweeps(sweeps):
    '''
    Find 1) the noise1 sweeps
         2) the noise2 sweeps
         4) all noise sweeps
    '''

    noise1_sweeps = get_sweeps_by_name(sweeps, NOISE1)
    noise2_sweeps = get_sweeps_by_name(sweeps, NOISE2)

    all_noise_sweeps = sorted(noise1_sweeps + noise2_sweeps, key=lambda x: x['sweep_number'])

    out = {
        'noise1': get_sweep_numbers(noise1_sweeps),
        'noise2': get_sweep_numbers(noise2_sweeps),
        'all_noise': get_sweep_numbers(all_noise_sweeps)
    }

    num_noise1_sweeps = len(noise1_sweeps)
    num_noise2_sweeps = len(noise2_sweeps)

    
    required_noise1_sweeps = 2
    required_noise2_sweeps = 2

    errors = []
    if num_noise1_sweeps < required_noise1_sweeps:
        errors.append("not enough noise1 sweeps (%d/%d)" % (num_noise1_sweeps, required_noise1_sweeps))
 
    if num_noise2_sweeps < required_noise2_sweeps:
        errors.append("not enough noise2 sweeps (%d/%d)" % (num_noise2_sweeps, required_noise2_sweeps))
         
    return out, errors


def find_optimization_sweeps(sweeps):

    sweep_index = { s['sweep_number']: s for s in sweeps }

    data = {}
    ssq_data, ssq_errs = find_short_square_sweeps(sweep_index)
    data.update(ssq_data)

    ramp_data, ramp_errs = find_ramp_sweeps(sweep_index)
    data.update(ramp_data)

    noise_data, noise_errs = find_noise_sweeps(sweep_index)
    data.update(noise_data)

    return data, sweep_index, ssq_errs + ramp_errs + noise_errs
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='find relevant sweeps from a sweep catalog')

    parser.add_argument('sweep_file', help='json file containing a list of sweeps for a cell')
    parser.add_argument('output_file', help='output json data config file')

    args = parser.parse_args()

    try:
        if not os.path.exists(args.sweep_file):
            raise Exception("sweep file (%s) does not exist" % args.sweep_file)

    except Exception, e:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = parse_arguments()

    sweeps = ju.read(args.sweep_file)

    data, sweep_index, errs = find_optimization_sweeps(sweeps)

    data["sweeps"] = sweep_index

    ju.write(args.output_file, data)

    sys.exit(len(errs) > 0)


if __name__ == "__main__":  main()
