import allensdk.core.json_utilities as ju
from allensdk.model.glif.threshold_adaptation import calc_spike_component_of_threshold_from_multiblip
import numpy as np

def test_calc_spike_component_of_threshold_from_muliblip():
	#TODO: note the sign of this decay const should change if change the abs value in the function
	multi_SS_strings=ju.read('data/476264255_multi_ss_dict.json')
	multi_SS={}
	multi_SS['current']= [np.array(v) for v in multi_SS_strings['current']]
	multi_SS['voltage']= [np.array(v) for v in multi_SS_strings['voltage']]
	
	const_to_add_to_thresh_for_reset, decay_const, thresh_inf=calc_spike_component_of_threshold_from_multiblip(multi_SS, 5e-05)
	if np.abs(const_to_add_to_thresh_for_reset-0.0057976216115)>1e-10:
		raise Exception("The const_to_add_to_thresh_for_reset is wrong")
	if np.abs(decay_const-73.1467930104)>1e-10:
		raise Exception("The decay_const is wrong")
	if np.abs(thresh_inf+0.052693268311)>1e-10:
		raise Exception("The thresh_inf is wrong")		
	

test_calc_spike_component_of_threshold_from_muliblip()