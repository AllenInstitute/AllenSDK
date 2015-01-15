import json
import warnings
from time import time

from neuron import GLIFNeuron
from experiment import GLIFExperiment
from preprocessor import GLIFPreprocessor
from optimizer import GLIFOptimizer
import utilities

import numpy as np

def read(data_config_file_name, model_config_file_name):
    data_config = None
    with open(data_config_file_name, 'rb') as f:
        data_config = json.loads(f.read())

    assert data_config is not None, Exception("Could not read data configuration file: %s" % data_config_file_name)

    model_config = {}
    with open(model_config_file_name, 'rb') as f:
        model_config = json.loads(f.read())

    return ConfigurationSetup(data_config=data_config,
                              neuron_config=model_config.get('neuron', None),
                              optimizer_config=model_config.get('optimizer', None),
                              preprocessing_config=model_config.get('preprocessing', None))

    raise Exception("could not read file: %s" % file_name)


class ConfigurationSetup( object ):
    def __init__(self, data_config, neuron_config, optimizer_config, preprocessing_config, 
                 spike_cutting_method=None, spike_determination_method='threshold'):

        self.data_config = data_config                 

        # configure the neuron
        self.neuron_config = neuron_config
        
        # initialize optimizer parameters
        self.optimizer_config = optimizer_config

        self.preprocessing_config = preprocessing_config

        self.spike_determination_method = spike_determination_method

    def write(self, model_config_file_name, data_config_file_name=None):
        with open(model_config_file_name, 'wb') as f:
            f.write(json.dumps(self.to_dict(), indent=2, default=utilities.json_handler))

        if data_config_file_name is not None:
            with open(data_config_file_name, 'wb') as f:
                f.write(json.dumps(self.data_config, indent=2))

    def to_dict(self):
        config = {
            'neuron': self.neuron_config,
            'optimizer': self.optimizer_config,
            'preprocessing_config': self.preprocessing_config
        }

        if self.neuron:
            config['neuron'].update(self.neuron.to_dict())

        if self.optimizer:
            config['optimizer'].update(self.optimizer.to_dict())
            
        return config


    def setup_neuron(self, neuron_config):
        '''this will load the correct neuron and configure it properly.  Right now there is only one type of neuron but 
        maybe one day there will be more.  This was called get_neuron in Tim's original experiment reader
        '''
        #TODO: note that Vr is not being set here but is calculated by the linear regression
        if neuron_config['type'] == GLIFNeuron.TYPE:
            self.neuron = GLIFNeuron(El=neuron_config['El'],
                                     dt=neuron_config['dt'],
                                     tau=neuron_config['tau'],
                                     R_input=neuron_config['R_input'],
                                     C=neuron_config['C'],
                                     asc_vector=neuron_config['asc_vector'],
                                     spike_cut_length=neuron_config['spike_cut_length'],
                                     th_inf=neuron_config['th_inf'],
                                     coeffs=neuron_config['coeffs'],
                                     AScurrent_dynamics_method=neuron_config['AScurrent_dynamics_method'],
                                     voltage_dynamics_method=neuron_config['voltage_dynamics_method'],
                                     threshold_dynamics_method=neuron_config['threshold_dynamics_method'],
                                     voltage_reset_method=neuron_config['voltage_reset_method'],
                                     AScurrent_reset_method=neuron_config['AScurrent_reset_method'],
                                     threshold_reset_method=neuron_config['threshold_reset_method'])
        else: 
            raise Exception("not implemented")        

        return self.neuron

    
    def setup_optimizer(self, optimize_stimulus_name, optimize_sweep_ids=None,
                        superthresh_blip_name='minimum_superthreshold_short_square', 
                        subthresh_blip_name='maximum_subthreshold_short_square', 
                        ramp_name='superthreshold_ramp',
                        all_noise_name='all_noise',
                        multi_square_name='multi_short_square'):

        # extract the sweeps to be used for optimization by name.  
        # If a subset of the sweeps are to be used, filter accordingly.
        optimize_sweeps = self.data_config[optimize_stimulus_name]

        if optimize_sweep_ids is not None:
            optimize_sweeps = [ optimize_sweeps[sid] for sid in optimize_sweep_ids ]

        # initialize the preprocessor
        prep = GLIFPreprocessor(self.neuron_config, 
                                self.data_config['filename'], 
                                self.data_config['sweeps'],
                                self.preprocessing_config)

        # preprocess the data (this will modify the neuron config)
        prep.preprocess_stimulus(optimize_sweeps,
                                 superthreshold_blip_sweeps=self.data_config.get(superthresh_blip_name, None),
                                 subthreshold_blip_sweeps=self.data_config.get(subthresh_blip_name, None),
                                 ramp_sweeps=self.data_config.get(ramp_name, None),
                                 all_noise_sweeps=self.data_config.get(all_noise_name, None),
                                 multi_blip_sweeps=self.data_config.get(multi_square_name, None),
                                 spike_determination_method=self.spike_determination_method)

        # set up the neuron based on the preprocessed neuron config
        self.setup_neuron(self.neuron_config)

        # initialize the experiment
        self.experiment = GLIFExperiment(neuron = self.neuron, 
                                         dt = self.neuron.dt,
                                         stim_list = prep.optimize_data['current'],
                                         grid_spike_index_target_list = prep.spike_ind_list,
                                         grid_spike_time_target_list = prep.grid_spike_time_target_list,
                                         interpolated_spike_time_target_list = prep.interpolated_spike_time_target_list,
                                         init_voltage = self.optimizer_config['init_voltage'],
                                         init_threshold = self.optimizer_config['init_threshold'],
                                         init_AScurrents = self.optimizer_config['init_AScurrents'],
                                         target_spike_mask = prep.target_spike_mask,
                                         param_fit_names = self.optimizer_config['param_fit_names'])  

        # initialize the optimizer
        self.optimizer = GLIFOptimizer(experiment = self.experiment, 
                                       dt = self.experiment.dt,              
                                       outer_iterations = self.optimizer_config['outer_iterations'],
                                       inner_iterations = self.optimizer_config['inner_iterations'],
                                       sigma_inner = self.optimizer_config['sigma_inner'],
                                       sigma_outer = self.optimizer_config['sigma_outer'],
                                       param_fit_names = self.optimizer_config['param_fit_names'],
                                       stim = prep.optimize_data['current'],
                                       error_function_name = self.optimizer_config['error_function'],
                                       neuron_num = self.optimizer_config['neuron_number'],
                                       xtol = self.optimizer_config['xtol'],
                                       ftol = self.optimizer_config['ftol'],
                                       internal_iterations = self.optimizer_config['internal_iterations'])

        return self.optimizer








