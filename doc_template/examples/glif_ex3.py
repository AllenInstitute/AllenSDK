import allensdk.core.json_utilities as json_utilities

from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.model.glif.simulate_neuron import simulate_neuron

neuron_config = json_utilities.read('472423251_neuron_config.json')
ephys_sweeps = json_utilities.read('ephys_sweeps.json')
ephys_file_name = '472423251.nwb'

neuron = GlifNeuron.from_dict(neuron_config)

simulate_neuron(neuron, ephys_sweeps, ephys_file_name, ephys_file_name, 0.05)
