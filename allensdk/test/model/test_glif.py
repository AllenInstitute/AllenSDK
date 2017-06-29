# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import pytest
from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.model.glif.simulate_neuron import simulate_neuron
from allensdk.core.nwb_data_set import NwbDataSet
import os
# import matplotlib.pyplot as plt

@pytest.fixture
def glif_api():
    endpoint = None

    if 'TEST_API_ENDPOINT' in os.environ:
        endpoint = os.environ['TEST_API_ENDPOINT']
        return GlifApi(endpoint)
    else:
        return GlifApi()

@pytest.fixture
def neuronal_model_id():
    neuronal_model_id = 566302806

    return neuronal_model_id


@pytest.fixture
def output():
    neuron_config = json_utilities.read('neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    ephys_file_name = 'stimulus.nwb'

    # pull out the stimulus for the first sweep
    ephys_sweep = ephys_sweeps[0]
    ds = NwbDataSet(ephys_file_name)
    data = ds.get_sweep(ephys_sweep['sweep_number'])
    stimulus = data['stimulus']

    # initialize the neuron
    # important! update the neuron's dt for your stimulus
    neuron = GlifNeuron.from_dict(neuron_config)
    neuron.dt = 1.0 / data['sampling_rate']

    # simulate the neuron
    truncate = 56041
    output = neuron.run(stimulus[0:truncate])

    return output


def test_1(glif_api, neuronal_model_id):
    glif_api.get_neuronal_model(neuronal_model_id)
    glif_api.cache_stimulus_file('stimulus.nwb')

    neuron_config = glif_api.get_neuron_config()
    json_utilities.write('neuron_config.json', neuron_config)

    ephys_sweeps = glif_api.get_ephys_sweeps()
    json_utilities.write('ephys_sweeps.json', ephys_sweeps)


def test_2():
    # initialize the neuron
    neuron_config = json_utilities.read('neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # make a short square pulse. stimulus units should be in Amps.
    stimulus = [0.0] * 100 + [10e-9] * 100 + [0.0] * 100

    # important! set the neuron's dt value for your stimulus in seconds
    neuron.dt = 5e-6

    # simulate the neuron
    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']
    spike_times = output['interpolated_spike_times']


@pytest.mark.skipif(True, reason="needs nwb file")
def test_3():
    neuron_config = json_utilities.read('neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    ephys_file_name = 'stimulus.nwb'

    neuron = GlifNeuron.from_dict(neuron_config)

    # sweep_numbers = [ s['sweep_number'] for s in ephys_sweeps
    #                  if s['stimulus_units'] == 'Amps' ]
    sweep_numbers = [7]
    simulate_neuron(neuron, sweep_numbers,
                    ephys_file_name, ephys_file_name, 0.05)


@pytest.fixture
def stimulus():
    neuron_config = json_utilities.read('neuron_config.json')
    ephys_sweeps = json_utilities.read('ephys_sweeps.json')
    ephys_file_name = 'stimulus.nwb'

    # pull out the stimulus for the first sweep
    ephys_sweep = ephys_sweeps[0]
    ds = NwbDataSet(ephys_file_name)
    data = ds.get_sweep(ephys_sweep['sweep_number'])
    stimulus = data['stimulus']

    return stimulus


@pytest.mark.skipif(True, reason="needs nwb file")
def test_4(output):
    voltage = output['voltage']
    threshold = output['threshold']
    spike_times = output['interpolated_spike_times']


@pytest.mark.skipif(True, reason="needs nwb file")
def test_5(output):
    voltage = output['voltage']
    threshold = output['threshold']
    interpolated_spike_times = output['interpolated_spike_times']
    spike_times = output['interpolated_spike_times']
    interpolated_spike_voltages = output['interpolated_spike_voltage']
    interpolated_spike_thresholds = output['interpolated_spike_threshold']
    grid_spike_indices = output['spike_time_steps']
    grid_spike_times = output['grid_spike_times']
    after_spike_currents = output['AScurrents']

#     # create a time array for plotting
#     time = np.arange(len(stimulus))*neuron.dt
#
#     plt.figure(figsize=(10, 10))
#
#     # plot stimulus
#     plt.subplot(3,1,1)
#     plt.plot(time, stimulus)
#     plt.xlabel('time (s)')
#     plt.ylabel('current (A)')
#     plt.title('Stimulus')
#
#     # plot model output
#     plt.subplot(3,1,2)
#     plt.plot(time,  voltage, label='voltage')
#     plt.plot(time,  threshold, label='threshold')
#
#     if grid_spike_indices:
#         plt.plot(interpolated_spike_times, interpolated_spike_voltages, 'x',
#                  label='interpolated spike')
#
#         plt.plot((grid_spike_indices-1)*neuron.dt, voltage[grid_spike_indices-1], '.',
#                  label='last step before spike')
#
#     plt.xlabel('time (s)')
#     plt.ylabel('voltage (V)')
#     plt.legend(loc=3)
#     plt.title('Model Response')
#
#     # plot after spike currents
#     plt.subplot(3,1,3)
#     for ii in range(np.shape(after_spike_currents)[1]):
#         plt.plot(time, after_spike_currents[:,ii])
#     plt.xlabel('time (s)')
#     plt.ylabel('current (A)')
#     plt.title('After Spike Currents')
#
#     plt.tight_layout()
#     plt.show()


@pytest.mark.skipif(True, reason="needs nwb file")
def test_6(stimulus):
    # define your own custom voltage reset rule
    # this one linearly scales the input voltage
    def custom_voltage_reset_rule(neuron, voltage_t0, custom_param_a, custom_param_b):
        return custom_param_a * voltage_t0 + custom_param_b

    # initialize a neuron from a neuron config file
    neuron_config = json_utilities.read('neuron_config.json')
    neuron = GlifNeuron.from_dict(neuron_config)

    # configure a new method and overwrite the neuron's old method
    method = neuron.configure_method('custom', custom_voltage_reset_rule,
                                     {'custom_param_a': 0.1, 'custom_param_b': 0.0})
    neuron.voltage_reset_method = method

    truncate = 56041
    output = neuron.run(stimulus[0:truncate])
