import sys, argparse, json, os, time
import allen_wrench.model.GLIF.configuration_setup as configuration_setup
from allen_wrench.core.orca_data_set import OrcaDataSet as EphysDataSet

def parse_arguments():
    parser = argparse.ArgumentParser(description='fit a neuron')

    parser.add_argument('--data_config_file', help='data configuration file name (sweeps properties, etc)', required=True)
    parser.add_argument('--model_config_file', help='configuration file output by optimizer', required=True)
    parser.add_argument('--output_file', help='output file name', required=True)
    parser.add_argument('--stimulus', help='stimulus type name', default=None)

    return parser.parse_args()

def valid_sweep(sweep):
    return 

def load_sweep(file_name, sweep_number):
    
    return data

def run_neuron(neuron, stimulus, init_voltage, init_threshold, init_AScurrents):
    start_time = time.time()
    print "simulating"

    (voltage, threshold, AScurrent_matrix, grid_spike_time, 
     interpolated_spike_time, grid_spike_index, interpolated_spike_voltage, 
     interpolated_spike_threshold) = neuron.run(init_voltage, 
                                                init_threshold,
                                                init_AScurrents,
                                                stimulus)    
    
    print "simulation time", time.time() - start_time
    
    return voltage

def main():
    args = parse_arguments()

    config = configuration_setup.read(args.data_config_file, args.model_config_file)
    neuron = config.setup_neuron(config.neuron_config)

    if args.stimulus is None:
        sweep_numbers = range(len(config.data_config['sweeps']))
    else:
        sweep_numbers = config.data_config[args.stimulus]

    file_name = config.data_config['filename']

    start_time = time.time()

    for sweep_number in sweep_numbers:
        load_start_time = time.time()
        print "loading sweep", sweep_number

        try:
            data = EphysDataSet(file_name).get_full_sweep(sweep_number)
            neuron.dt = 1.0 / data['sampling_rate']
        except Exception,e:
            print "Failed to load sweep, skipping.", e
            continue

        print "load time", time.time() - load_start_time

        voltage = run_neuron(neuron, data['stimulus'], 
                             config.optimizer_config['init_voltage'], 
                             config.optimizer_config['init_threshold'],
                             config.optimizer_config['init_AScurrents'])

        print "writing"
        write_start_time = time.time()

        out_data = EphysDataSet(args.output_file).set_full_sweep(sweep_number, stimulus=None, response=voltage)

        print "write time", time.time() - write_start_time

    print "total elapsed time", time.time() - start_time

if __name__ == "__main__": main()
