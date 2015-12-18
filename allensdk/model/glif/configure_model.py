#going to need to take preprocessed dictionaries and model configuration and create 
#a preprocessed model configuration and preprocessed_config file
#
#something will have to tell it what parameters to take out of the preprocessed dict
import os
import sys
import logging
import time
import numpy as np
import argparse

import allensdk.core.json_utilities as ju

import find_sweeps as fs

class ModelConfigurationException( Exception ): pass

DEFAULT_NEURON_PARAMETERS = {
    "type": "GLIF", 
    "dt": 5e-05, 
    "El": 0,
    "asc_tau_array": [ 1, 1 ],
    "asc_amp_array": [ 0, 0 ],
    "init_AScurrents": [ 0.0, 0.0 ],
    "init_threshold": 0.02, 
    "init_voltage": 0.0,
    "extrapolation_method_name": "endpoints",
    "dt_multiplier": 1
    }


DEFAULT_OPTIMIZER_PARAMETERS = { 
    "xtol": 1e-05, 
    "ftol": 1e-05, 
    "sigma_outer": 0.3, 
    "sigma_inner": 0.01, 
    "inner_iterations": 3,  
    "outer_iterations": 3, 
    "internal_iterations": 10000000,
    "iteration_info": [],
    "param_fit_names": [],
    "cut": 0,
    "bessel": { 'N': 4, 'Wn': .1 }
    }
    

def validate_method_requirements(method_config_name, has_rheo, has_mss):
    if not has_mss and not has_rheo:
        valid_configs = ['LIF', 'LIF_ASC']  #these levels are always run since before something else figured out thi
    elif has_mss and not has_rheo:
        valid_configs = ['LIF', 'LIF_ASC','LIF_R', 'LIF_R_ASC', 'LIF_R_ASC_AT']
    elif not has_mss and has_rheo:
        valid_configs = ['LIF', 'LIF_ASC','LIF_PWL', 'LIF_ASC_PWL']
    elif has_mss and has_rheo:
        valid_configs = ['LIF', 'LIF_ASC', 'LIF_R', 'LIF_R_ASC', 'LIF_R_AT', 'LIF_R_ASC_AT', 'LIF_PWL', 'LIF_ASC_PWL', 'LIF_R_PWL', 'LIF_R_ASC_PWL', 'LIF_R_AT_PWL', 'LIF_R_ASC_AT_PWL']

    if method_config_name not in valid_configs:
        raise ModelConfigurationException("Model type %s cannot be configured due to missing data (rheo: %s, mss: %s)" % ( method_config_name, str(has_rheo), str(has_mss) ))


def update_neuron_method(method_type, arg_method_name, neuron_config):
    neuron_config[method_type] = { 'name': arg_method_name, 'params': None }


def configure_model(method_config, preprocessor_values):
    neuron_config = {}
    neuron_config.update(DEFAULT_NEURON_PARAMETERS)
    optimizer_config = {}
    optimizer_config.update(DEFAULT_OPTIMIZER_PARAMETERS)

    #a) select values want to use out of the preprocessor_values via specifying parameter_gropus
    #b) look what levels are available via the levels available in the preprocessor_values.
    #--not sure whether adapting threshold is necessary for level 6 maybe we should add some levels

    #The sweeps need to pass both the sweep properties and have a value because the computation using the 
    #available stimulus actually is able to be calculated
    
    # Skip trace if subthreshold noise has a spike in it.
    noise1_ind =  [ n1i for n1i in preprocessor_values['spike_inds']['noise1'] if n1i is not None ]
    noise1_ind = np.concatenate(noise1_ind)
    if np.any(noise1_ind * preprocessor_values['dt'] < 8.0):
        raise ModelConfigurationException("Subthreshold region of noise1 stimulus contains spikes.")

    # check if there is a short square triple
    if preprocessor_values['threshold_adaptation']['b_spike_component_of_threshold'] and preprocessor_values['threshold_adaptation']['a_spike_component_of_threshold']:
        has_mss=True
    else:
        has_mss=False

    # check if there is ramp to rheo 
    if ( ( preprocessor_values['nonlinearity_parameters']['line_param_ElV_all'] is not None ) and 
         ( preprocessor_values['nonlinearity_parameters']['line_param_RV_all'] is not None ) ):
        has_rheo=True
    else: 
        has_rheo=False
        
    # make sure that the requested method config meets minimum requirements
    validate_method_requirements(method_config['name'], has_rheo, has_mss)

    update_neuron_method('AScurrent_dynamics_method', method_config['AScurrent_dynamics_method'], neuron_config)
    update_neuron_method('voltage_dynamics_method', method_config['voltage_dynamics_method'], neuron_config)
    update_neuron_method('threshold_dynamics_method', method_config['threshold_dynamics_method'], neuron_config)
    update_neuron_method('AScurrent_reset_method', method_config['AScurrent_reset_method'], neuron_config)
    update_neuron_method('voltage_reset_method', method_config['voltage_reset_method'], neuron_config)
    update_neuron_method('threshold_reset_method', method_config['threshold_reset_method'], neuron_config)

    neuron_config['El_reference'] = preprocessor_values['El_reference']
    neuron_config['C'] = preprocessor_values['C']           
    neuron_config['El'] = preprocessor_values['El']  
    neuron_config['spike_cut_length'] = preprocessor_values['spike_cut_length']
    neuron_config['asc_amp_array'] = preprocessor_values['asc_amp_array']
    neuron_config['asc_tau_array'] = preprocessor_values['asc_tau_array']
    neuron_config['R_input'] = preprocessor_values['R_input']
    neuron_config['th_inf'] = preprocessor_values['th_inf']
    neuron_config['th_adapt'] = preprocessor_values['th_adapt']

    optimizer_config['error_function'] = method_config['error_function']
    optimizer_config['param_fit_names'] = method_config['param_fit_names']
    optimizer_config['nwb_file'] = preprocessor_values['nwb_file']

    #b) choose the sets want from the preprocessor_values 
    #TODO: currently the model configuration just alters things in the neuron config which is kind of annoying
    #T0DO: make sure that if it is already being written above it is not being rewritten below
    configure_method_parameters(                  
        neuron_config,
        optimizer_config,
        preprocessor_values['spike_cutting_slope'],
        preprocessor_values['spike_cutting_intercept'],
        #!!!!!!!!MAKE SURE THESE ARE IMPLIMENTED CORRECTLY
        preprocessor_values['nonlinearity_parameters']['line_param_RV_all'], 
        preprocessor_values['nonlinearity_parameters']['line_param_ElV_all'],
        #!!!!!!!!MAKE SURE THESE ARE IMPLIMENTED CORRECTY
        preprocessor_values['threshold_adaptation']['a_spike_component_of_threshold'],
        preprocessor_values['threshold_adaptation']['b_spike_component_of_threshold'],
        preprocessor_values['threshold_adaptation']['a_voltage_component_of_threshold'],
        preprocessor_values['threshold_adaptation']['b_voltage_component_of_threshold'],
        preprocessor_values['MLIN']['var_of_section'],
        preprocessor_values['MLIN']['sv_for_expsymm'],
        preprocessor_values['MLIN']['tau_from_AC'])

    return {
        'neuron': neuron_config,
        'optimizer': optimizer_config
        }

#The idea here should be to put in things that are needed
#rename things as appropriate
def configure_method_parameters(neuron_config,
                                optimizer_config,
                                slope_at_min_expVar_list,
                                intercept_at_min_expVar_list,
                                #!!!!!!!!MAKE SURE THESE ARE IMPLIMENTED CORRECTLY
                                pwR, 
                                pwEl,
                                #!!!!!!!!MAKE SURE THESE ARE IMPLIMENTED CORRECTY
                                a_spike_component_of_threshold,
                                b_spike_component_of_threshold,
                                a_voltage_component_of_threshold,
                                b_voltage_component_of_threshold,
                                var_of_section,
                                sv_for_expsymm,
                                tau_from_AC):
    
    # initialize the generic method data 
    neuron_config['init_method_data'] = {}
    
    
    # configure voltage reset rules
    method_config = neuron_config['voltage_reset_method']
    if method_config.get('params', None) is None:
        if method_config['name'] == 'zero':
            method_config['params'] = {}
        elif method_config['name'] == 'v_before':                        
            method_config['params'] = {
                'a': slope_at_min_expVar_list,
                'b': intercept_at_min_expVar_list
                }
    
        elif method_config['name'] == 'i_v_before':
            method_config['params'] = {
                'a': 1,
                'b': 2,
                'c': 3
                }
            raise ModelConfigurationException('i_v_before of voltage reset method is not yet implemented')
        elif method_config['name'] == 'fixed':
            raise ModelConfigurationException('cannot use fixed voltage reset method in preprocessor')
        else:
            method_config['params'] = {}
    
    # configure threshold reset rules
    
    method_config = neuron_config['threshold_reset_method']        
    if method_config.get('params', None) is None:
        coeff_th_inf = neuron_config.get('coeffs', {}).get('th_inf',1.0)
        adjusted_th_inf = neuron_config['th_inf'] * coeff_th_inf
            
        if method_config['name'] == 'max_v_th':
            raise ModelConfigurationException('max_v_th threshold reset rule is not currently in use')
            
        elif method_config['name'] == 'th_before':
            raise ModelConfigurationException('th_before is not currently in use')
            
        elif method_config['name'] == 'inf':
            method_config['params'] = {}
            neuron_config['init_threshold'] = adjusted_th_inf
                
        elif method_config['name'] == 'sum_spike_and_adapt':
            method_config['params'] = {}
            neuron_config['init_threshold'] = adjusted_th_inf
                
        elif method_config['name'] == 'fixed':
            raise ModelConfigurationException("cannot use fixed threshold reset method in preprocessor")
        else:  
            raise ModelConfigurationException("unknown threshold reset method: ", method_config['name'])
    
    # configure voltage dynamics rules
    
    method_config = neuron_config['voltage_dynamics_method']
    if method_config.get('params', None) is None:
        if method_config['name'] == 'quadratic_i_of_v':
            raise ModelConfigurationException('quadraticIofV of voltage_dynamics_method preprocessing is not yet implemented')
        elif method_config['name'] == 'forward_euler_linear':
            method_config['params'] = {}
        elif method_config['name'] == 'euler_exact_linear':
            method_config['params'] = {}
        elif method_config['name'] == 'forward_euler_piecewise':
            #TODO: corinnet MAKE SURE THESE ARE CORRECT! NUMBERS AND THE WAY THEY ARE BEING IMPLIMENTED IN NEURON METHODS 
            method_config['params'] = {
                'R_tlparam1': pwR[0],
                'R_tlparam2': pwR[1],
                'R_t1param3': pwR[2], 
                'El_slope_param': pwEl[0],
                'El_intercept_param':pwEl[1]
                # 'El_tlparam1': pwEl[0],
                # 'El_tlparam2': pwEl[1], 
                # 'El_t1param3': pwEl[2]
                }
        elif method_config['name'] == 'euler_exact_piecewise':
            #TODO: corinnet MAKE SURE THESE ARE CORRECT! NUMBERS AND THE WAY THEY ARE BEING IMPLIMENTED IN NEURON METHODS 
            method_config['params'] = {
                'R_tlparam1': pwR[0],
                'R_tlparam2': pwR[1],
                'R_t1param3': pwR[2], 
                'El_slope_param': pwEl[0],
                'El_intercept_param':pwEl[1]
                # 'El_tlparam1': pwEl[0],
                # 'El_tlparam2': pwEl[1], 
                # 'El_t1param3': pwEl[2]
                }

                #TODO: corinnet THESE ARE SKETCHY CHECK THIS THIS USED TO USE TWO_LINES
#                print 'rtest', max_of_line_and_const(0, method_config['params']['R_tlparam1'], method_config['params']['R_tlparam2'], method_config['params']['R_t1param3'])
#                print 'rtest .03', max_of_line_and_const(0.03, method_config['params']['R_tlparam1'], method_config['params']['R_tlparam2'], method_config['params']['R_t1param3'])
#                print 'El test 0', -max_of_line_and_const(0, method_config['params']['El_tlparam1'], method_config['params']['El_tlparam2'], method_config['params']['El_t1param3'])  
#                print 'El test .03', -max_of_line_and_const(0.03, method_config['params']['El_tlparam1'], method_config['params']['El_tlparam2'], method_config['params']['El_t1param3'])
        else:
            raise ModelConfigurationException("unknown voltage dynamics method: ", method_config['name'])
    
    # configure threshold dynamics rules
    
    method_config = neuron_config['threshold_dynamics_method']
    
    if method_config.get('params', None) is None:
        if method_config['name'] == 'adapt_standard':
            raise ModelConfigurationException('adapt_standard is not currently used as a threshold_dynamics method' )
            method_config['params'] = {
                #TODO: switch out a and b for amp and decay of exponential
                'a': a_spike_component_of_threshold,
                'b': b_spike_component_of_threshold 
                }
            
        elif method_config['name'] == 'sum_spike_and_adapt':
            neuron_config['init_method_data'].update({
                    'a_spike': a_spike_component_of_threshold,
                    'b_spike': b_spike_component_of_threshold,
                    'a_voltage': a_voltage_component_of_threshold,
                    'b_voltage': b_voltage_component_of_threshold
                    })
    
            method_config['params'] = {}
    
        elif method_config['name'] == 'spike_component':
            neuron_config['init_method_data'].update({
                    'a_spike': a_spike_component_of_threshold,
                    'b_spike': b_spike_component_of_threshold,
                    'a_voltage': 0,
                    'b_voltage': 0
                    })
    
            method_config['params'] = {}
    
        elif method_config['name'] == 'inf':
            method_config['params'] = {}
    
        elif method_config['name'] == 'fixed':
            raise ModelConfigurationException('cannot use fixed threshold dynamics method in preprocessor')
    
        else:
            raise ModelConfigurationException("unknown threshold dynamics method: ", method_config['name'])
    
    # configure ascurrent dynamics rules
    method_config = neuron_config['AScurrent_dynamics_method']
    if method_config.get('params', None) is None:
        # TODO: rename 'vector' to something more specific
        if method_config['name'] == 'vector': 
            method_config['params'] = {
                'vector': [1, 2, 3]
                }
            raise ModelConfigurationException('vector of AScurrent_dynamics_method is not yet implemented')  
        elif method_config['name'] == 'none':
            method_config['params'] = {}
        elif method_config['name'] == 'exp':
            method_config['params'] = {}
        else:
            raise ModelConfigurationException("unknown AScurrent dynamics method: ", method_config['name'])
    
    # configure ascurrent reset rule 
    # this is down here because it depends on numbers computed for the AScurrent_dynamics_method
    
    method_config = neuron_config['AScurrent_reset_method']
    if method_config.get('params', None) is None:
        if method_config['name'] == 'sum':
            method_config['params'] = {
                'r': np.ones(len(neuron_config['asc_tau_array']))
                }
        elif method_config['name'] == 'none':
            method_config['params'] = {}
        else:
            raise ModelConfigurationException("unknown AScurrent reset method: ", method_config['name'])
    
    #--if using the MLIN error function calculate the cdf
    if optimizer_config['error_function']=='MLIN':
        optimizer_config['error_function_data'] = {
            'subthreshold_long_square_voltage_variance': var_of_section,
            'sv_for_expsymm': sv_for_expsymm,
            'tau_from_AC': tau_from_AC
            }


    # validation

    # make sure that the initial ascurrents have the correct size
    if len(neuron_config['init_AScurrents']) != len(neuron_config['asc_tau_array']):
        raise ModelConfigurationException("init_AScurrents have incorrect length.")
    
    spike_cut_length = neuron_config.get('spike_cut_length', None)
    
    if spike_cut_length is None:
        raise ModelConfigurationException("Spike cut length must be set, but it is not.")
    
    if spike_cut_length < 0:
        raise ModelConfigurationException("Spike cut length must be non-negative.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preprocessor_values_path', help='path to preprocessor values json')
    parser.add_argument('method_config_path', help='path to method configuration json')
    parser.add_argument('output_path', help='path to store final model configuration')

    args = parser.parse_args()

    preprocessor_values = ju.read(args.preprocessor_values_path)
    method_config = ju.read(args.method_config_path)

    out_config = configure_model(method_config, preprocessor_values)

    ju.write(out_config_path, out_config)


if __name__ == "__main__": main()
