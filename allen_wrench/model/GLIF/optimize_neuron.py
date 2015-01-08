import sys, argparse, json, os
import allen_wrench.model.GLIF.configuration_setup as configuration_setup

DEFAULT_STIMULUS = 'noise1_run1'
DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_model_config.json')

def iteration_finished_callback(optimizer, outer, inner):
    print 'finished outer iteration: ', outer, 'inner iteration: ', inner
    print optimizer.iteration_info[-1]

def parse_arguments():
    parser = argparse.ArgumentParser(description='fit a neuron')

    parser.add_argument('--data_config_file', help='data configuration file name (sweeps properties, etc)', required=True)
    parser.add_argument('--method_config_file', help='method configuration file name (reset and dynamics rules)', required=True)
    parser.add_argument('--output_config_file', help='output file name', required=True)

    parser.add_argument('--model_config_file', help='model configuration file name (preprocessing defaults, neuron defaults, etc)', default=DEFAULT_MODEL_CONFIG)
    parser.add_argument('--stimulus', help='stimulus type name', default=DEFAULT_STIMULUS)

    return parser.parse_args()

def main():
    args = parse_arguments()

    config = configuration_setup.read(args.data_config_file, args.model_config_file)

    with open(args.method_config_file, 'rb') as f:
        method_config = json.loads(f.read())
        
    config.neuron_config['AScurrent_dynamics_method'] = { 'name': method_config['AScurrent_dynamics_method'], 'params': None }
    config.neuron_config['voltage_dynamics_method'] =   { 'name': method_config['voltage_dynamics_method'],   'params': None }
    config.neuron_config['threshold_dynamics_method'] = { 'name': method_config['threshold_dynamics_method'], 'params': None }
    config.neuron_config['AScurrent_reset_method'] =    { 'name': method_config['AScurrent_reset_method'],    'params': None }
    config.neuron_config['voltage_reset_method'] =      { 'name': method_config['voltage_reset_method'],      'params': None }
    config.neuron_config['threshold_reset_method'] =    { 'name': method_config['threshold_reset_method'],    'params': None }

    optimizer = config.setup_optimizer(args.stimulus)

    best_params, begin_params = optimizer.run_many(config.optimizer_config['outer_loop'], iteration_finished_callback) 
    
    print 'finished optimizing'
    print 'initial params', begin_params
    print 'best params', best_params

    config.write(args.output_config_file)

if __name__ == "__main__": main()

