from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities

neuronal_model_id = 472423251

glif_api = GlifApi()
glif_api.get_neuronal_model(neuronal_model_id)
glif_api.cache_stimulus_file('stimulus.nwb')

neuron_config = glif_api.get_neuron_config()
json_utilities.write('neuron_config.json', neuron_config)

ephys_sweeps = glif_api.get_ephys_sweeps()
json_utilities.write('ephys_sweeps.json', ephys_sweeps)
