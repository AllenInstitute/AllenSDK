import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron
from allensdk.model.glif.simulate_neuron import simulate_neuron
import os, shutil, logging

#NEURONAL_MODEL_ID = 491547163 # level 1 LIF
NEURONAL_MODEL_ID = 491547171 # level 5 GLIF

OUTPUT_DIR = 'tmp'

def test_download():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    os.makedirs(OUTPUT_DIR)

    glif_api = GlifApi()
    glif_api.get_neuronal_model(NEURONAL_MODEL_ID)
    glif_api.cache_stimulus_file(os.path.join(OUTPUT_DIR, '%d.nwb' % NEURONAL_MODEL_ID))

    neuron_config = glif_api.get_neuron_config()
    json_utilities.write(os.path.join(OUTPUT_DIR, '%d_neuron_config.json' % NEURONAL_MODEL_ID), neuron_config)

    ephys_sweeps = glif_api.get_ephys_sweeps()
    json_utilities.write(os.path.join(OUTPUT_DIR, 'ephys_sweeps.json'), ephys_sweeps)

def test_run():
    import allensdk.core.json_utilities as json_utilities


    # initialize the neuron
    neuron_config = json_utilities.read(os.path.join(OUTPUT_DIR, '%d_neuron_config.json' % NEURONAL_MODEL_ID))
    neuron = GlifNeuron.from_dict(neuron_config)

    # make a short square pulse. stimulus units should be in Amps.
    stimulus = [ 0.0 ] * 100 + [ 10e-9 ] * 100 + [ 0.0 ] * 100

    # important! set the neuron's dt value for your stimulus in seconds
    neuron.dt = 5e-6

    # simulate the neuron
    output = neuron.run(stimulus)

    voltage = output['voltage']
    threshold = output['threshold']

    plt.plot(voltage)
    plt.plot(threshold)
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot.png'))

def test_simulate():
    logging.getLogger().setLevel(logging.DEBUG)
    neuron_config = json_utilities.read(os.path.join(OUTPUT_DIR, '%d_neuron_config.json' % NEURONAL_MODEL_ID))
    ephys_sweeps = json_utilities.read(os.path.join(OUTPUT_DIR, 'ephys_sweeps.json'))
    ephys_file_name = os.path.join(OUTPUT_DIR, '%d.nwb' % NEURONAL_MODEL_ID)

    neuron = GlifNeuron.from_dict(neuron_config)

    sweep_numbers = [ s['sweep_number'] for s in ephys_sweeps ]
    simulate_neuron(neuron, sweep_numbers, ephys_file_name, ephys_file_name, 0.05)

if __name__ == "__main__": 
    #test_download()
    #test_run()
    test_simulate()
