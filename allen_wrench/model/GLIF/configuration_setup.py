import json
import warnings
from time import time

from neuron import GLIFNeuron
from experiment import GLIFExperiment
from preprocessor import GLIFPreprocessor
from optimizer import GLIFOptimizer

import numpy as np


def json_handler(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        raise TypeError, 'Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj))


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
    '''things that need to be read in at the moment should be what is being fit and the number of k's
    '''
    def __init__(self, data_config, neuron_config=None, optimizer_config=None,
                 save_fig=True, spike_cutting_method=None, preprocessing_config=None, spike_determination_method='threshold'):

        self.data_config = data_config                 

        # configure the neuron
        self.neuron_config = neuron_config
        if neuron_config is None:
            self.neuron_config = dict(ConfigurationSetup.DEFAULT_NEURON_CONFIG)

        # initialize optimizer parameters
        self.optimizer_config = optimizer_config
        if self.optimizer_config is None:
            self.optimizer_config = dict(ConfigurationSetup.DEFAULT_OPTIMIZER_CONFIG)

#         self.spike_cutting_method = spike_cutting_method
#         if spike_cutting_method is None:
#             self.spike_cutting_method = {'specifiedTime': .004}
#        self.spikeCuttingMethod_dict={'None':[]}
#        self.spikeCuttingMethod_dict={'specifiedTime': .0074}  #note that this value has to be smaller than the smallest ISI.  There is a built in error function that will check thi
    
        self.preprocessing_config = preprocessing_config
        if self.preprocessing_config is None:
            self.preprocessing_config = dict(ConfigurationSetup.DEFAULT_PREPROCESSING_CONFIG)
#         self.preprocessing_methods = preprocessing_methods
#         if self.preprocessing_methods is None:
#             self.preprocessing_methods = {'None':[]}
#        self.dictOfPreprocessMethods = kwargs.get('preprocessing_methods', )
#        self.dictOfPreprocessMethods={'subSample':{'present_time_step': self.neuron.dt, 'desired_time_step': 0.01},
#                                      'cut_extra_current':[20000:40000]}
#        self.dictOfPreprocessMethods={'cut_extra_current':[], 'zeroOutElViaInjCurrent':{'blip_index':[2], 'blip_ind':[], 'input_resistance':[] }}  #NEED TO MAKE SURE THIS IS CORRECT FOR THE DATA YOU ARE PROCESSING
#        self.dictOfPreprocessMethods={'cut_extra_current':[]}

        self.save_fig = save_fig

        self.spike_determination_method = spike_determination_method

    def write(self, model_config_file_name, data_config_file_name=None):
        with open(model_config_file_name, 'wb') as f:
            f.write(json.dumps(self.to_dict(), indent=2, default=json_handler))

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
                                     Rinput=neuron_config['Rinput'],
                                     Cap=neuron_config['Cap'],
                                     a_vector=neuron_config['a_vector'],
                                     spike_cut_length=neuron_config['spike_cut_length'],
                                     th_inf=neuron_config['th_inf'],
                                     
                                     coeff_C=neuron_config['coeff_C'],
                                     coeff_G=neuron_config['coeff_G'],
                                     coeff_b=neuron_config['coeff_b'],
                                     coeff_a=neuron_config['coeff_a'],
                                     coeff_a_vector=neuron_config['coeff_a_vector'],
                                     
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
                        all_noise_name='all_noise'):

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
                                 spike_determination_method=self.spike_determination_method)

        # set up the neuron based on the preprocessed neuron config
        self.setup_neuron(self.neuron_config)

        # make sure that the initial ascurrents have the correct size
        init_AScurrents = self.optimizer_config.get('init_AScurrents', [])
        if len(init_AScurrents) != len(self.neuron.a_vector):
            warnings.warn('ConfigurationSetup thinks the init_AScurrents have incorrect length.  Setting to zeros.')
            self.optimizer_config['init_AScurrents'] = np.zeros(len(self.neuron.a_vector))

        # make sure that the optimizer parameter bounds have the correct size
        lower_bounds = self.optimizer_config.get('param_lower_bounds', [])
        upper_bounds = self.optimizer_config.get('param_upper_bounds', [])
        correct_num_bounds = len(self.optimizer_config['param_fit_names']) + len(self.neuron.a_vector) - 1
        
        if len(lower_bounds) != correct_num_bounds:
            warnings.warn('ConfigurationSetup thinks the param_lower_bounds has incorrect length.  Setting to zeros.')
            self.optimizer_config['param_lower_bounds'] = np.zeros(correct_num_bounds)

        if len(upper_bounds) != correct_num_bounds:
            warnings.warn('ConfigurationSetup thinks the param_upper_bounds has incorrect length.  Setting to ones.')
            self.optimizer_config['param_upper_bounds'] = np.ones(correct_num_bounds)

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
                                         fit_names_list = self.optimizer_config['param_fit_names'])  

        # initialize the optimizer
        self.optimizer = GLIFOptimizer(experiment = self.experiment, 
                                       dt = self.experiment.dt,              
                                       inner_loop = self.optimizer_config['inner_loop'],
                                       start_time = time(),
                                       save_file_name = self.optimizer_config['save_file_name'],
                                       stim = prep.optimize_data['current'],
                                       lower_bounds = np.array(self.optimizer_config['param_lower_bounds']), 
                                       upper_bounds = np.array(self.optimizer_config['param_upper_bounds']), #!!!!!!!!!THIS WI4LL HAVE TO BE ADAPTED FOR VARIOUS PARAMETERS!!!!!!!!!!!
                                       eps = self.optimizer_config['eps'],
                                       param_fit_names = self.optimizer_config['param_fit_names'],
                                       error_function_name = self.optimizer_config['error_function'],
                                       neuron_num = self.optimizer_config['neuron_number'],
                                       xtol = self.optimizer_config['xtol'],
                                       ftol = self.optimizer_config['ftol'],
                                       internal_iterations = self.optimizer_config['internal_iterations'],
                                       internal_func = self.optimizer_config['internal_func'])

        return self.optimizer


    DEFAULT_OPTIMIZER_CONFIG = {
        'test_hold_out': True,
        'single_hold_out_repeat': False,
        'outer_loop': 1,
        'inner_loop': 1,
        'neuron_number': 1,
        'error_function': 'VSD',
        'xtol': 0.0000001,
        'ftol': 0.0000001,
        'internal_iterations': 1,
        'internal_func': 1000000,
        'param_fit_names': ['coeff_a', 'coeff_a_vector'],
        'param_upper_bounds': [1,1,1,1,1],
        'param_lower_bounds': [0,0,0,0,0],
        'save_file_name': 'default_optimizer_file_name.json',
        'eps': 0.01,
        'init_voltage': 0.0,
        'init_threshold': 0.02,
        'init_AScurrents': [0,0,0,0]  #this intentially set to a list--probably because that would be how it comes in in a potential configuration file?
    }

    DEFAULT_NEURON_CONFIG = {
        'tau': [0.01, 0.04, 0.16, 0.64],
        'El': 0,
        'dt':.00005,
        'Rinput': 159e6,  
        'Cap': 143.58e-12,
        'ER1': 0.07, #not used
        'ER2': -0.02, #not used
        'coeff_C': 1.0,
        'coeff_G': 1.0,
        'coeff_a': 1.0,
        'coeff_b': 1.0,
        'Vr': 0.004,
        'th_inf':0.027,
        'a_vector': [ 1e-12, 1e-12, 1e-12, 1e-12 ],
        'coeff_a_vector': [ 1, -1, 1, -1 ],
        'spike_cut_length': 0, #86 #int(.0043/neuron_config["dt"])}
        'AScurrent_dynamics_method': {'name': 'exp','params': {} },
        #'AScurrent_dynamics_method': { 'name': 'fixed','params': { 'value': 10 } }, # made up value
        #'AScurrent_dynamics_method': { 'name': 'vector': [ 1, 1, 1, 1 ] }, # made up value
        'voltage_dynamics_method': { 'name': 'linear','params': {} },
        #'voltage_dynamics_method': {'name': 'quadraticIofV','params': { a:1.10111713e-12, b:4.33940334e-09, c:-1.33483359e-07, d:.015, e:3.8e-11 }},
        'threshold_dynamics_method': {'name': 'fixed','params': { 'value': 10 }},
        #'threshold_dynamics_method': {'name': 'adapt_standard','params': { 'a': 1, 'b': 10 }},
        'voltage_reset_method': { 'name': 'Vbefore','params': {'a': 1.0516, 'b': 0.0051},
        "b": 0.0051 },
#        'voltage_reset_method': { 'name': 'Vbefore','params': { 'a': 1.0516, 'b': 0.0051 }},
        #'voltage_reset_method': {'name': 'IandVbefore','params': {} }, 
        #'voltage_reset_method': {'name': 'fixed','params': { 'value': 0.003 }}, #was 0 for shiyong 
        'threshold_reset_method': {'name': 'fixed','params': { 'value': 0.010 }} #it was .01 for shiyong
        #'threshold_reset_method': {'name': 'from_paper','params': { 'th_reset': 0.010 }} #it was .01 for shiyong
    }
    
    DEFAULT_PREPROCESSING_CONFIG = {
        'subsample': { 'desired_time_step': 0.00005 },
        'cut_extra_current': False,
        'zer_out_el_via_inj_current': False,
    }






