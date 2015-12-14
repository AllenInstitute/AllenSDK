import logging
from preprocess_neuron import preprocess_neuron

import allensdk.core.json_utilities as ju

def test_preprocess_neuron():
    logging.getLogger().setLevel(logging.DEBUG)
    data_config_file = "/data/mat/Corinne/GLIF_subset/data_config_files/329552531_data_config.json"
    test_data = "/data/mat/Corinne/GLIF_subset/preprocessed_dicts/dictionaries/329552531_preprocessed_dict.json"

    data_config = ju.read(data_config_file)
    nwb_file = data_config["filename"]
    sweep_list = data_config["sweeps"].values()
    dt = 5e-05
    cut = 0
    bessel = { 'N': 4, 'Wn': .1 }

    d = preprocess_neuron(nwb_file, sweep_list, dt, cut, bessel)
    print d
    

if __name__ == "__main__": test_preprocess_neuron()
