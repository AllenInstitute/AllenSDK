# Copyright 2015-2016 Allen Institute for Brain Science
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

import logging
import time
import argparse
import os
import numpy as np
import allensdk.core.json_utilities as json_utilities
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.api.queries.glif_api import GlifApi
from allensdk.model.glif.glif_neuron import GlifNeuron

DEFAULT_SPIKE_CUT_VALUE = 0.05 # 50mV

def parse_arguments():
    ''' Use argparse to get required arguments from the command line '''
    parser = argparse.ArgumentParser(description='fit a neuron')

    parser.add_argument('--ephys_file', help='ephys file name')
    parser.add_argument('--sweeps_file', help='JSON file listing sweep properties')
    parser.add_argument('--neuron_config_file', help='neuron configuration JSON file ')
    parser.add_argument('--neuronal_model_id', help='id of the neuronal model. Used when downloading sweep properties.', type=int)
    parser.add_argument('--output_ephys_file', help='output file name')
    parser.add_argument('--log_level', help='log level', default=logging.INFO)
    parser.add_argument('--spike_cut_value', help='value to fill in for spike duration', default=DEFAULT_SPIKE_CUT_VALUE, type=float)

    return parser.parse_args()


def simulate_sweep(neuron, stimulus, spike_cut_value):
    ''' Simulate a neuron given a stimulus and initial conditions. '''

    start_time = time.time()

    logging.debug("simulating")

    data = neuron.run(stimulus)

    voltage = data['voltage']
    voltage[np.isnan(voltage)] = spike_cut_value

    logging.debug("simulation time %f" % (time.time() - start_time))

    return data


def load_sweep(file_name, sweep_number):
    ''' Load the stimulus for a sweep from file. '''
    logging.debug("loading sweep %d" % sweep_number)

    load_start_time = time.time()
    data = NwbDataSet(file_name).get_sweep(sweep_number)

    logging.debug("load time %f" % (time.time() - load_start_time))

    return data


def write_sweep_response(file_name, sweep_number, response, spike_times):
    ''' Overwrite the response in a file. '''

    logging.debug("writing sweep")

    write_start_time = time.time()
    ephds = NwbDataSet(file_name)

    ephds.set_sweep(sweep_number, stimulus=None, response=response)
    ephds.set_spike_times(sweep_number, spike_times)

    logging.debug("write time %f" % (time.time() - write_start_time))


def simulate_sweep_from_file(neuron, sweep_number, input_file_name, output_file_name, spike_cut_value):
    ''' Load a sweep stimulus, simulate the response, and write it out. '''

    sweep_start_time = time.time()

    try:
        data = load_sweep(input_file_name, sweep_number)
    except Exception as e:
        logging.warning("Failed to load sweep, skipping. (%s)" % str(e))
        raise

        # tell the neuron what dt should be for this sweep
    neuron.dt = 1.0 / data['sampling_rate']

    sim_data = simulate_sweep(neuron, data['stimulus'], spike_cut_value)

    write_sweep_response(output_file_name, sweep_number, sim_data['voltage'], sim_data['interpolated_spike_times'])

    logging.debug("total sweep time %f" % ( time.time() - sweep_start_time ))

def simulate_neuron(neuron, sweep_numbers, input_file_name, output_file_name, spike_cut_value):

    start_time = time.time()

    for sweep_number in sweep_numbers:
        simulate_sweep_from_file(neuron, sweep_number, input_file_name, output_file_name, spike_cut_value)

    logging.debug("total elapsed time %f" % (time.time() - start_time))

def main():
    args = parse_arguments()

    logging.getLogger().setLevel(args.log_level)

    glif_api = None
    if (args.neuron_config_file is None or
        args.sweeps_file is None or
        args.ephys_file is None):

        assert args.neuronal_model_id is not None, Exception("A neuronal model id is required if no neuron config file, sweeps file, or ephys data file is provided.")

        glif_api = GlifApi()
        glif_api.get_neuronal_model(args.neuronal_model_id)

    if args.neuron_config_file:
        neuron_config = json_utilities.read(args.neuron_config_file)
    else:
        neuron_config = glif_api.get_neuron_config()

    if args.sweeps_file:
        sweeps = json_utilities.read(args.sweeps_file)
    else:
        sweeps = glif_api.get_ephys_sweeps()

    if args.ephys_file:
        ephys_file = args.ephys_file
    else:
        ephys_file = 'stimulus_%d.nwb' % args.neuronal_model_id

        if not os.path.exists(ephys_file):
            logging.info("Downloading stimulus to %s." % ephys_file)
            glif_api.cache_stimulus_file(ephys_file)
        else:
            logging.warning("Reusing %s because it already exists." % ephys_file)

    if args.output_ephys_file:
        output_ephys_file = args.output_ephys_file
    else:
        logging.warning("Overwriting input file data with simulated data in place.")
        output_ephys_file = ephys_file


    neuron = GlifNeuron.from_dict(neuron_config)

    # filter out test sweeps
    sweep_numbers = [ s['sweep_number'] for s in sweeps if s['stimulus_name'] != 'Test' ]

    simulate_neuron(neuron, sweep_numbers, ephys_file, output_ephys_file, args.spike_cut_value)



if __name__ == "__main__": main()
