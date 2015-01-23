import logging, time
import sys, argparse, json, os
import allen_wrench.model.GLIF.configuration_setup as configuration_setup

DEFAULT_STIMULUS = 'noise1_run1'
DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_model_config.json')

def parse_arguments():
    ''' Use argparse to get required arguments from the command line '''

    parser = argparse.ArgumentParser(description='fit a neuron')

    parser.add_argument('--data_config_file', help='data configuration file name (sweeps properties, etc)', required=True)
    parser.add_argument('--output_config_file', help='output file name', required=True)
    parser.add_argument('--model_config_file', help='model configuration file name (preprocessing defaults, neuron defaults, etc)', default=DEFAULT_MODEL_CONFIG)
    parser.add_argument('--stimulus', help='stimulus type name', default=DEFAULT_STIMULUS)
    parser.add_argument('--log_level', help='log_level', default=logging.INFO)

    return parser.parse_args()


def iteration_finished_callback(optimizer, outer, inner):
    ''' Print some debug information when the optimizer finishes an iteration '''
    logging.info('finished outer iteration: %d, inner iteration: %d' % (outer, inner))
    logging.info(repr(optimizer.iteration_info[-1]))


def optimize_neuron(optimizer, iteration_finished_callback):
    ''' Run a GLIF optimizer for a given number of iterations '''

    start_time = time.time()

    best_params, begin_params = optimizer.run_many(iteration_finished_callback) 

    logging.debug("optimize time %f" % (time.time() - start_time))

    logging.info('finished optimizing')
    logging.info('initial params ' + str(begin_params))
    logging.info('best params' +  str(best_params))


def main():
    args = parse_arguments()
    logging.getLogger().setLevel(args.log_level)

    config = configuration_setup.read(args.data_config_file, args.model_config_file)
    optimizer = config.setup_optimizer(args.stimulus)
    optimize_neuron(optimizer, iteration_finished_callback)

    config.write(args.output_config_file)


if __name__ == "__main__": main()

