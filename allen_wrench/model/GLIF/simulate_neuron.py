import logging, time
import sys, argparse, json, os

import allen_wrench.model.GLIF.configuration_setup as configuration_setup
from allen_wrench.core.orca_data_set import OrcaDataSet as EphysDataSet

logger = logging.getLogger()


def parse_arguments():
    ''' Use argparse to get required arguments from the command line '''
    parser = argparse.ArgumentParser(description='fit a neuron')

    parser.add_argument('--data_config_file', help='data configuration file name (sweeps properties, etc)', required=True)
    parser.add_argument('--model_config_file', help='configuration file output by optimizer', required=True)
    parser.add_argument('--output_file', help='output file name', required=True)
    parser.add_argument('--stimulus', help='stimulus type name', default=None)

    return parser.parse_args()


def simulate_sweep(neuron, stimulus, init_voltage, init_threshold, init_AScurrents):
    ''' Simulate a neuron given a stimulus and initial conditions. '''

    start_time = time.time()

    logger.debug("simulating")

    (voltage, threshold, AScurrent_matrix, grid_spike_time, 
     interpolated_spike_time, grid_spike_index, interpolated_spike_voltage, 
     interpolated_spike_threshold) = neuron.run(init_voltage, 
                                                init_threshold,
                                                init_AScurrents,
                                                stimulus)    
    
    logger.debug("simulation time %f" % (time.time() - start_time))
    
    return voltage


def load_sweep_stimulus(file_name, sweep_number):
    ''' Load the stimulus for a sweep from file. '''
    logger.debug("loading sweep %d" % sweep_number)
    
    load_start_time = time.time()
    data = EphysDataSet(file_name).get_full_sweep(sweep_number)

    logger.debug("load time %f" + (time.time() - load_start_time))

    return data['stimulus']


def write_sweep_response(file_name, sweep_number, response):
    ''' Overwrite the response in a file. '''

    logger.debug("writing sweep")

    write_start_time = time.time()
    out_data = EphysDataSet(args.output_file).set_full_sweep(sweep_number, stimulus=None, response=voltage)
    
    logger.debug("write time %f" % (time.time() - write_start_time))

    
def simulate_sweep_from_file(neuron, sweep_number, input_file_name, output_file_name, neuron, init_voltage, init_threshold, init_AScurrents):
    ''' Load a sweep stimulus, simulate the response, and write it out. '''
    
    sweep_start_time = time.time()
    
    try:
        stimulus = load_sweep_stimulus(input_file_name, sweep_number)
    except Exception,e:
        logger.warning("Failed to load sweep, skipping. " + str(e))
        return
        
        # tell the neuron what dt should be for this sweep
    neuron.dt = 1.0 / data['sampling_rate']
    
    response = simulate_sweep(neuron, stimulus, init_voltage, init_threshold, init_AScurrents)
    
    write_sweep_response(output_file_name, sweep_number, response)
    
    logger.debug("total sweep time %f", ( time.time() - sweep_start_time)
                 
def main():
    args = parse_arguments()
                 
    config = configuration_setup.read(args.data_config_file, args.model_config_file)
    neuron = config.setup_neuron(config.neuron_config)

    if args.stimulus is None:
        sweep_numbers = range(len(config.data_config['sweeps']))
    else:
        sweep_numbers = config.data_config[args.stimulus]

    input_file_name = config.data_config['filename']

    start_time = time.time()

    for sweep_number in sweep_numbers:
        simulate_sweep_from_file(neuron, sweep_number, 
                                 input_file_name, args.output_file,
                                 config.optimizer_config['init_voltage'], 
                                 config.optimizer_config['init_threshold'],
                                 config.optimizer_config['init_AScurrents'])
                 

    logger.debug("total elapsed time %f" % (time.time() - start_time))

if __name__ == "__main__": main()
