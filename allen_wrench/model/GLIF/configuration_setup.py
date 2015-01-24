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
    data_config = utilities.read_json(data_config_file_name)
    model_config = utilities.read_json(model_config_file_name)

    return ConfigurationSetup(data_config=data_config,
                              neuron_config=model_config.get('neuron', None),
                              optimizer_config=model_config.get('optimizer', None),
                              preprocessor_config=model_config.get('preprocessor', None))

    raise Exception("could not read file: %s" % file_name)


class ConfigurationSetup( object ):
    MINIMUM_SUPERTHRESHOLD_SHORT_SQUARE = 'minimum_superthreshold_short_square'
    MAXIMUM_SUBTHRESHOLD_SHORT_SQUARE = 'maximum_subthreshold_short_square'
    SUPERTHRESHOLD_RAMP = 'superthreshold_ramp'
    ALL_NOISE = 'all_noise'
    SHORT_SQUARE_TRIPLE = 'short_square_triple'

    def __init__(self, data_config, neuron_config, optimizer_config, preprocessor_config, 
                 spike_cutting_method=None, spike_determination_method='threshold'):

        self.data_config = data_config                 
        self.neuron_config = neuron_config
        self.optimizer_config = optimizer_config
        self.preprocessor_config = preprocessor_config

        self.neuron = None
        self.optimizer = None
        self.preprocessor = None
        self.experiment = None

        self.spike_determination_method = spike_determination_method

    def write(self, model_config_file_name, data_config_file_name=None):
        utilities.write_json(model_config_file_name, self.to_dict())

        if data_config_file_name is not None:
            utilities.write_json(data_config_file_name, self.data_config)

    def to_dict(self):
        config = {
            'neuron': self.neuron_config,
            'optimizer': self.optimizer_config,
            'preprocessor': self.preprocessor_config
        }

        if self.neuron:
            config['neuron'].update(self.neuron.to_dict())

        if self.optimizer:
            config['optimizer'].update(self.optimizer.to_dict())
        
        if self.preprocessor:
            config['preprocessor'].update(self.preprocessor.to_dict())
            
        return config


    def setup_neuron(self):
        '''this will load the correct neuron and configure it properly.  Right now there is only one type of neuron but 
        maybe one day there will be more.  This was called get_neuron in Tim's original experiment reader
        '''
        
        #TODO: note that Vr is not being set here but is calculated by the linear regression
        if self.neuron_config['type'] == GLIFNeuron.TYPE:
            self.neuron = GLIFNeuron(El=self.neuron_config['El'],
                                     dt=self.neuron_config['dt'],
                                     tau=self.neuron_config['tau'],
                                     R_input=self.neuron_config['R_input'],
                                     C=self.neuron_config['C'],
                                     asc_vector=self.neuron_config['asc_vector'],
                                     spike_cut_length=self.neuron_config['spike_cut_length'],
                                     th_inf=self.neuron_config['th_inf'],
                                     coeffs=self.neuron_config.get('coeffs', {}),
                                     AScurrent_dynamics_method=self.neuron_config['AScurrent_dynamics_method'],
                                     voltage_dynamics_method=self.neuron_config['voltage_dynamics_method'],
                                     threshold_dynamics_method=self.neuron_config['threshold_dynamics_method'],
                                     voltage_reset_method=self.neuron_config['voltage_reset_method'],
                                     AScurrent_reset_method=self.neuron_config['AScurrent_reset_method'],
                                     threshold_reset_method=self.neuron_config['threshold_reset_method'])
        else: 
            raise Exception("not implemented")        

        return self.neuron

    def get_sweeps(self, stimulus_name, sweep_ids=None):
        # extract the sweeps to be used for optimization by name.  
        # If a subset of the sweeps are to be used, filter accordingly.
        sweeps = self.data_config[stimulus_name]

        if sweep_ids is not None:
            sweeps = [ sweeps[sid] for sid in sweep_ids ]

        return sweeps

    def setup_preprocessor(self):
        self.preprocessor = GLIFPreprocessor(self.neuron_config, 
                                             self.optimizer_config,
                                             self.data_config['filename'], 
                                             self.data_config['sweeps'],
                                             self.preprocessor_config.get('spike_time_steps',None),
                                             self.preprocessor_config.get('interpolated_spike_times',None),
                                             self.preprocessor_config.get('grid_spike_times', None),
                                             self.preprocessor_config.get('target_spike_mask', None),
                                             self.preprocessor_config.get('interpolated_spike_voltages',None),
                                             self.preprocessor_config.get('grid_spike_voltages', None),
                                             self.preprocessor_config.get('optional_methods',{}))

        return self.preprocessor

    def run_preprocessor(self, optimize_stimulus_name, optimize_sweep_ids=None,
                         superthresh_ssq_name=MINIMUM_SUPERTHRESHOLD_SHORT_SQUARE, 
                         subthresh_ssq_name=MAXIMUM_SUBTHRESHOLD_SHORT_SQUARE,
                         ramp_name=SUPERTHRESHOLD_RAMP,
                         all_noise_name=ALL_NOISE,
                         ssq_triple_name=SHORT_SQUARE_TRIPLE,
                         force_preprocessing=False):
        ''' initialize the preprocessor (if necessary) and run it '''

        if not self.preprocessor:
            self.setup_preprocessor()

        optimize_sweeps = self.get_sweeps(optimize_stimulus_name, optimize_sweep_ids)

        # if the preprocessor has a spike index list, no need to preprocess
        if not self.preprocessor.ready() or force_preprocessing:

            # preprocess the data (this will modify the neuron config)
            self.preprocessor.preprocess_stimulus(optimize_sweeps,
                                                  superthresh_ssq_sweeps=self.data_config.get(superthresh_ssq_name, None),
                                                  subthresh_ssq_sweeps=self.data_config.get(subthresh_ssq_name, None),
                                                  ramp_sweeps=self.data_config.get(ramp_name, None),
                                                  all_noise_sweeps=self.data_config.get(all_noise_name, None),
                                                  ssq_triple_sweeps=self.data_config.get(ssq_triple_name, None),
                                                  spike_determination_method=self.spike_determination_method)
        return self.preprocessor
                                             

        
    def setup_optimizer(self, optimize_stimulus_name, optimize_sweep_ids=None,
                        superthresh_ssq_name=MINIMUM_SUPERTHRESHOLD_SHORT_SQUARE, 
                        subthresh_ssq_name=MAXIMUM_SUBTHRESHOLD_SHORT_SQUARE,
                        ramp_name=SUPERTHRESHOLD_RAMP,
                        all_noise_name=ALL_NOISE,
                        ssq_triple_name=SHORT_SQUARE_TRIPLE,
                        force_preprocessing=False):

        self.run_preprocessor(optimize_stimulus_name, optimize_sweep_ids,
                              superthresh_ssq_name,  subthresh_ssq_name,
                              ramp_name,
                              all_noise_name,
                              ssq_triple_name,
                              force_preprocessing)

        self.setup_neuron()

        optimize_sweeps = self.get_sweeps(optimize_stimulus_name, optimize_sweep_ids)

        # if the preprocessor has already run, it will have the data we want to optimize already loaded,
        # we need to load the data appropriately if the preprocessor hasn't run, though.
        optimize_data = self.preprocessor.optimize_data

        if not optimize_data:
            optimize_data = self.preprocessor.load_stimulus(self.data_config['filename'], optimize_sweeps)

        # initialize the experiment

        self.experiment = GLIFExperiment(neuron = self.neuron, 
                                         dt = self.neuron.dt,
                                         stim_list = optimize_data['current'],
                                         spike_time_steps = self.preprocessor.spike_time_steps,
                                         grid_spike_times = self.preprocessor.grid_spike_times,
                                         interpolated_spike_times = self.preprocessor.interpolated_spike_times,
                                         grid_spike_voltages = self.preprocessor.grid_spike_voltages,
                                         interpolated_spike_voltages = self.preprocessor.interpolated_spike_voltages,
                                         init_voltage = self.optimizer_config['init_voltage'],
                                         init_threshold = self.optimizer_config['init_threshold'],
                                         init_AScurrents = self.optimizer_config['init_AScurrents'],
                                         target_spike_mask = self.preprocessor.target_spike_mask,
                                         param_fit_names = self.optimizer_config['param_fit_names'])  

        # initialize the optimizer
        self.optimizer = GLIFOptimizer(experiment = self.experiment, 
                                       dt = self.neuron.dt,              
                                       outer_iterations = self.optimizer_config['outer_iterations'],
                                       inner_iterations = self.optimizer_config['inner_iterations'],
                                       sigma_inner = self.optimizer_config['sigma_inner'],
                                       sigma_outer = self.optimizer_config['sigma_outer'],
                                       param_fit_names = self.optimizer_config['param_fit_names'],
                                       stim = optimize_data['current'],
                                       error_function_name = self.optimizer_config['error_function'],
                                       xtol = self.optimizer_config['xtol'],
                                       ftol = self.optimizer_config['ftol'],
                                       internal_iterations = self.optimizer_config['internal_iterations'])

        return self.optimizer

