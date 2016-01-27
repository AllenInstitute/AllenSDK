from allensdk.model.glif.threshold_adaptation import find_multiblip_spikes
import allensdk.core.json_utilities as ju

import numpy as np

def test_find_multiblip_spikes():
    multi_SS=ju.read('data/476264255_multi_ss_dict.json')
    multi_SS_v=[np.array(v) for v in multi_SS['voltage']]
    multi_SS_i=[np.array(v) for v in multi_SS['current']]
    spike_ind = find_multiblip_spikes(multi_SS_i, multi_SS_v, 5e-05)
    
    test_data = [np.array([40599]), np.array([40467, 40732]), np.array([40464, 40679]), np.array([40464, 40764]), np.array([40465, 40844]), np.array([40465, 40925]), np.array([40467, 41546]), np.array([40464, 41087]), np.array([40464, 41869]), np.array([40465, 40732]), np.array([40469, 40887]), np.array([40467, 40759]), np.array([40464, 40845]), np.array([40465, 40924]), np.array([40467, 41546]), np.array([40464]), np.array([40464, 41166]), np.array([40464, 40731]), np.array([40459, 40678]), np.array([40459, 40759]), np.array([40464, 40844]), np.array([40464, 40925]), np.array([40464, 41539]), np.array([40464, 41705]), np.array([40464])]
    assert len(test_data) == len(spike_ind), 'oops, different lengths'
    for i in xrange(len(test_data)):
        assert not np.any(spike_ind[i] != test_data[i]), "the multiblip spike detector is not working correctly"

test_find_multiblip_spikes()
