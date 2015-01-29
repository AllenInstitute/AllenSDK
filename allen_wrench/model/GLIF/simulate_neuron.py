import logging, time
import sys, argparse, json, os
import numpy as np

import allen_wrench.model.GLIF.utilities as utilities

from allen_wrench.model.GLIF.neuron import GLIFNeuron
from allen_wrench.core.orca_data_set import OrcaDataSet as EphysDataSet

def parse_arguments():
    ''' Use argparse to get required arguments from the command line '''
    parser = argparse.ArgumentParser(description='fit a neuron')

    parser.add_argument('--ephys_file', help='ephys file name', required=True)
    parser.add_argument('--sweeps_file', help='JSON file listing sweep properties')
    parser.add_argument('--ephys_result_id', help='id of the ephys result, used when downloading sweep properties')
    parser.add_argument('--neuron_config_file', help='neuron configuration JSON file ', required=True)
    parser.add_argument('--output_ephys_file', help='output file name', required=True)
    parser.add_argument('--log_level', help='log_level', default=logging.INFO)

    return parser.parse_args()


def simulate_sweep(neuron, stimulus, init_voltage, init_threshold, init_AScurrents):
    ''' Simulate a neuron given a stimulus and initial conditions. '''

    start_time = time.time()

    logging.debug("simulating")

    data = neuron.run(init_voltage, 
                      init_threshold,
                      init_AScurrents,
                      stimulus)    
    
    logging.debug("simulation time %f" % (time.time() - start_time))
    
    return data


def load_sweep(file_name, sweep_number):
    ''' Load the stimulus for a sweep from file. '''
    logging.debug("loading sweep %d" % sweep_number)
    
    load_start_time = time.time()
    data = EphysDataSet(file_name).get_sweep(sweep_number)

    logging.debug("load time %f" % (time.time() - load_start_time))

    return data


def write_sweep_response(file_name, sweep_number, response, spike_times):
    ''' Overwrite the response in a file. '''

    logging.debug("writing sweep")

    write_start_time = time.time()
    ephds = EphysDataSet(file_name)
    
    ephds.set_sweep(sweep_number, stimulus=None, response=response)
    ephds.set_spike_times(sweep_number, spike_times)
    
    logging.debug("write time %f" % (time.time() - write_start_time))

    
def simulate_sweep_from_file(neuron, sweep_number, input_file_name, output_file_name, init_voltage, init_threshold, init_AScurrents):
    ''' Load a sweep stimulus, simulate the response, and write it out. '''
    
    sweep_start_time = time.time()
    
    try:
        data = load_sweep(input_file_name, sweep_number)
    except Exception,e:
        logging.warning("Failed to load sweep, skipping. (%s)" % str(e))
        raise
        
        # tell the neuron what dt should be for this sweep
    neuron.dt = 1.0 / data['sampling_rate']
    
    sim_data = simulate_sweep(neuron, data['stimulus'], init_voltage, init_threshold, init_AScurrents)

    write_sweep_response(output_file_name, sweep_number, sim_data['voltage'], sim_data['interpolated_spike_times'])

    logging.debug("total sweep time %f" % ( time.time() - sweep_start_time ))

def simulate_neuron(neuron, sweeps, input_file_name, output_file_name, 
                    init_voltage = None, init_threshold = None, init_AScurrents = None):

    start_time = time.time()

    filtered_sweeps = [ sweep for sweep in sweeps if sweep['ephys_stimulus']['ephys_stimulus_type']['name'] != 'Unknown' ]

    if init_voltage is None:
        init_voltage = 0.0

    if init_threshold is None:
        init_threshold = 0.0

    if init_AScurrents is None:
        init_AScurrents = np.zeros(len(neuron.tau))

    for sweep in filtered_sweeps:
        simulate_sweep_from_file(neuron, sweep['sweep_number'], 
                                 input_file_name, output_file_name,
                                 init_voltage,
                                 init_threshold,
                                 init_AScurrents)
                 
    logging.debug("total elapsed time %f" % (time.time() - start_time))    

def main():
    args = parse_arguments()

    logging.getLogger().setLevel(args.log_level)
    
    neuron_config = utilities.read_json(args.neuron_config_file)

    if args.sweeps_file:
        sweeps = utilities.read_json(args.sweeps_file)
    elif args.ephys_result_id:
        #sweeps = api.download_ephys_sweeps(args.ephys_result_id)
        raise Exception("TODO: Code for downloading sweeps from the API still needs to be written")
    else:
        raise Exception("No sweeps file provided. Provide an ephys_result_id to download sweep metadata automatically.")

    neuron = GLIFNeuron.from_dict(neuron_config)

    simulate_neuron(neuron, sweeps, args.ephys_file, args.output_ephys_file) 



if __name__ == "__main__": main()
