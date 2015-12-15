import argparse, sys

import allensdk.core.json_utilities as ju
import allensdk.model.glif.find_sweeps as fs

from allensdk.model.glif.glif_optimizer_neuron import GlifOptimizerNeuron
from allensdk.model.glif.glif_experiment import GlifExperiment
from allensdk.model.glif.glif_optimizer import GlifOptimizer

from preprocess_neuron import load_sweeps
from find_spikes import find_spikes_list

def get_optimize_sweep_numbers(sweep_index):
    return = fs.get_noise_sweeps(sweep_index)['noise1']

def optimize_neuron(model_config, sweep_index):
    neuron_config = model_config['neuron']
    optimizer_config = model_config['optimizer']

    neuron = GlifOptimizerNeuron.from_dict(neuron_config)

    optimize_sweeps = get_optimize_sweep_numbers(sweep_index)
    optimize_data = load_sweeps(optimizer_config['nwb_file'], optimize_sweeps)

    # offset all voltages by El_reference
    El_reference = neuron_config['El_reference']

    spike_ind, spike_v = find_spikes_list(optimize_data['voltage'], neuron_config['dt'])
    
    grid_spike_times=[ st*neuron_config['dt'] for st in spike_ind ]
    grid_spike_voltages_in_ref_to_zero = [ v[si] - El_reference for v,si in zip(spike_v, spike_ind)]

    resp_list = [ d - El_reference for d in optimize_data['voltage'] ]  

    experiment = GlifExperiment(neuron = neuron, 
                                dt = neuron.dt,
                                stim_list = optimize_data['current'],
                                resp_list = resp_list,
                                spike_time_steps = spike_ind, 
                                grid_spike_times = grid_spike_times,
                                grid_spike_voltages = grid_spike_voltages_in_ref_to_zero,
                                param_fit_names = optimizer_config['param_fit_names'))


    optimizer = GlifOptimizer(experiment = experiment, 
                              dt = neuron.dt,              
                              outer_iterations = optimizer_config['outer_iterations'],
                              inner_iterations = optimizer_config['inner_iterations'],
                              sigma_inner = optimizer_config['sigma_inner'],
                              sigma_outer = optimizer_config['sigma_outer'],
                              param_fit_names = optimizer_config['param_fit_names'],
                              stim = optimize_data['current'],
                              error_function_name = optimizer_config['error_function'],
                              error_function_data = optimizer_config['error_function_data'],
                              xtol = optimizer_config['xtol'],
                              ftol = optimizer_config['ftol'],
                              internal_iterations = optimizer_config['internal_iterations'],
                              init_params = optimizer_config.get('init_params', None))

    def save(optimizer, outer, inner):
        print 'finished outer', outer, 'inner', inner
        config.write(optimized_model_config_file_name)
    
    best_param, begin_param = optimizer.run_many(save) 
    
    return best_param, begin_param

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sweeps_file')
    parser.add_argument('output_file')
    parser.add_argument("--dt", default=DEFAULT_DT)
    parser.add_argument("--bessel", default=DEFAULT_BESSEL)
    parser.add_argument("--cut", default=DEFAULT_CUT)

    args = parser.parse_args()

    model_config = ju.read(args.model_config_file)
    sweep_list = ju.read(args.sweeps_file)

    sweep_index = { s['sweep_number']:s for s in sweep_list }

    try:
        output = optimize_neuron(model_config, sweep_index, dt, cut, bessel)
        ju.write(args.output_file, output)
    except Exception, e:
        logging.error(e.message)

if __name__ == "__main__": main()
