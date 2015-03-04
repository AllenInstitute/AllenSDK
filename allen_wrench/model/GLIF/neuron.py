import logging

import numpy as np
import copy
import warnings
import json, utilities

from neuron_methods import GLIFNeuronMethod, METHOD_LIBRARY


class GLIFNeuronException( Exception ):
    """ A simple exception for catching simulation errors and reporting intermediate data. """
    def __init__(self, message, data):
        super(Exception, self).__init__(message)
        self.data = data
    
class GLIFNeuron( object ):    
    """ Implements the current-based Mihalas Neiber GLIF neuron.  Simulations model the voltage, 
    threshold, and afterspike currents of a neuron given an input stimulus.  A set of modular dynamics
    rules are applied until voltage crosses threshold, at which point a set of modular reset rules are 
    applied. See neuron_methods.py for a list of what options there are for voltage, threshold, and
    afterspike current dynamics and reset rules.
    """

    TYPE = "GLIF"

    def __init__(self, El, dt, tau, R_input, C, asc_vector, spike_cut_length, th_inf, coeffs,
                 AScurrent_dynamics_method, voltage_dynamics_method, threshold_dynamics_method,
                 AScurrent_reset_method, voltage_reset_method, threshold_reset_method,
                 init_voltage, init_threshold, init_AScurrents): 
        """ Initialize the neuron.

        :parameter El: resting potential 
        :type El: float
        :parameter dt: duration between time steps
        :type dt: float
        :parameter tau: 
        :type tau:
        :parameter R_input: input resistance
        :type R_input: float
        :parameter C: capacitance
        :type C: float
        :parameter asc_vector: afterspike current vector.  one element per element of tau.
        :type asc_vector: np.ndarray
        :parameter spike_cut_length: how many time steps to replace with NaNs when a spike occurs.
        :type spike_cut_length: int
        :parameter th_inf: instantaneous threshold
        :type th_inf: float
        :parameter coeffs: dictionary coefficients premultiplied to neuron properties during simulation. used for optimization.
        :type coeffs: dict
        :parameter AScurrent_dynamics_method: dictionary containing the 'name' of the afterspike current dynamics method to use and a 'params' dictionary parameters to pass to that function.
        :type AScurrent_dynamics_method: dict
        :parameter voltage_dynamics_method: dictionary containing the 'name' of the voltage dynamics method to use and a 'params' dictionary parameters to pass to that function.
        :type voltage_dynamics_method: dict
        :parameter threshold_dynamics_method: dictionary containing the 'name' of the threshold dynamics method to use and a 'params' dictionary parameters to pass to that function.
        :type threshold_dynamics_method: dict
        :parameter AScurrent_reset_method: dictionary containing the 'name' of the afterspike current dynamics method to use and a 'params' dictionary parameters to pass to that function.
        :type AScurrent_reset_method: dict
        :parameter voltage_reset_method: dictionary containing the 'name' of the voltage dynamics method to use and a 'params' dictionary parameters to pass to that function.
        :type voltage_reset_method: dict
        :parameter threshold_reset_method: dictionary containing the 'name' of the threshold dynamics method to use and a 'params' dictionary parameters to pass to that function.
        :type threshold_reset_method: dict
        :parameter init_voltage: initial voltage value
        :type init_voltage: float
        :parameter init_threshold: initiali spike threshold value
        :type init_threshold: float
        :parameter init_AScurrents: initial afterspike current vector. one element per element of tau.
        :type init_AScurrents: np.ndarray
        """

        self.type = GLIFNeuron.TYPE
        self.El = El
        self.dt = dt
        self.tau = np.array(tau)
        self.R_input = R_input
        self.C = C
        self.asc_vector = np.array(asc_vector)
        self.spike_cut_length = int(spike_cut_length)
        self.th_inf = th_inf
        
        self.init_voltage = init_voltage
        self.init_threshold = init_threshold
        self.init_AScurrents = init_AScurrents

        assert len(tau) == len(asc_vector), Exception("After-spike current vector must have same length as tau (%d vs %d)" % (asc_vector, tau))
        assert len(self.init_AScurrents) == len(self.tau), Exception("init_AScurrents length (%d) must have same length as tau (%d)" % (len(self.init_AScurrents), len(self.tau)))


        # values computed based on inputs
        self.k = 1.0 / self.tau
        self.G = 1.0 / self.R_input

        # Values that can be fit: They scale the input values.  
        # These are allowed to have default values because they are going to get optimized.
        self.coeffs = {
            'th_inf': 1,
            'C': 1,
            'G': 1,
            'b': 1,
            'a': 1,
            'asc_vector': np.ones(len(self.tau))
        }

        self.coeffs.update(coeffs)
        
        logging.debug('spike cut length: %d' %  self.spike_cut_length)

        # initialize dynamics methods
        self.AScurrent_dynamics_method = self.configure_library_method('AScurrent_dynamics_method', AScurrent_dynamics_method)
        self.voltage_dynamics_method = self.configure_library_method('voltage_dynamics_method', voltage_dynamics_method)
        self.threshold_dynamics_method = self.configure_library_method('threshold_dynamics_method', threshold_dynamics_method)

        # initialize reset methods
        self.AScurrent_reset_method = self.configure_library_method('AScurrent_reset_method', AScurrent_reset_method)
        self.voltage_reset_method = self.configure_library_method('voltage_reset_method', voltage_reset_method)
        self.threshold_reset_method = self.configure_library_method('threshold_reset_method', threshold_reset_method)

    def __str__(self):
        return json.dumps(self.to_dict(), default=utilities.json_handler, indent=2)

    @staticmethod
    def from_dict(d):
        return GLIFNeuron(El = d['El'],
                          dt = d['dt'],
                          tau = d['tau'],
                          R_input = d['R_input'],
                          C = d['C'],
                          asc_vector = d['asc_vector'],
                          spike_cut_length = d['spike_cut_length'],
                          th_inf = d['th_inf'],
                          coeffs = d.get('coeffs', {}),
                          AScurrent_dynamics_method = d['AScurrent_dynamics_method'],
                          voltage_dynamics_method = d['voltage_dynamics_method'],
                          threshold_dynamics_method = d['threshold_dynamics_method'],
                          voltage_reset_method = d['voltage_reset_method'],
                          AScurrent_reset_method = d['AScurrent_reset_method'],
                          threshold_reset_method = d['threshold_reset_method'],
                          init_voltage = d['init_voltage'],
                          init_threshold = d['init_threshold'],
                          init_AScurrents = d['init_AScurrents'])

    def to_dict(self):
        """ Convert the neuron to a serializable dictionary. """
        return {
            'type': self.type,
            'El': self.El,
            'dt': self.dt,
            'tau': self.tau,
            'R_input': self.R_input,
            'C': self.C,
            'asc_vector': self.asc_vector,
            'spike_cut_length': self.spike_cut_length,
            'th_inf': self.th_inf,
            'coeffs': self.coeffs,
            'AScurrent_dynamics_method': self.AScurrent_dynamics_method,
            'voltage_dynamics_method': self.voltage_dynamics_method,
            'threshold_dynamics_method': self.threshold_dynamics_method,
            'AScurrent_reset_method': self.AScurrent_reset_method,
            'voltage_reset_method': self.voltage_reset_method,
            'threshold_reset_method': self.threshold_reset_method,
            'init_voltage': self.init_voltage,
            'init_threshold': self.init_threshold,
            'init_AScurrents': self.init_AScurrents
        }

    @staticmethod
    def configure_method(method_name, method, method_params):
        """ Create a GLIFNeuronMethod instance given a name, a function, and function parameters. 
        This is just a shortcut to the GLIFNeuronMethod constructor.

        :parameter method_name: name for refering to this method later
        :type method_name: string
        :parameter method: a python function 
        :type method: function
        :parameter method_parameters: function arguments whose values should be fixed
        :type method: dict
        :returns: a GLIFNeuronMethod instance
        :rtype: GLIFNeuronMethod
        """ 

        return GLIFNeuronMethod(method_name, method, method_params)

    @staticmethod
    def configure_library_method(method_type, params):
        """ Create a GLIFNeuronMethod instance out of a library of functions organized by type name. 
        This refers to the METHOD_LIBRARY in neuron_methods.py, which lays out the available functions 
        that can be used for dynamics and reset rules.

        :parameter method_type: the name of a function category (e.g. 'AScurrent_dynamics_method' for the afterspike current dynamics methods)
        :type method_type: string
        :parameter params: a dictionary with two members. 'name': the string name of function you want, and 'params': parameters you want to pass to that function
        :type params: dict
        :returns: a GLIFNeuronMethod instance
        :rtype: GLIFNeuronMethod
        """
        method_options = METHOD_LIBRARY.get(method_type, None)

        assert method_options is not None, Exception("Unknown method type (%s)" % method_type)
        
        method_name = params.get('name', None)
        method_params = params.get('params', None)
        
        assert method_name is not None, Exception("Method configuration for %s has no 'name'" % (method_type))
        assert method_params is not None, Exception("Method configuration for %s has no 'params'" % (method_params))
        
        method = method_options.get(method_name, None)
        
        assert method is not None, Exception("unknown method name %s of type %s" % (method_name, method_type))
        
        return GLIFNeuron.configure_method(method_name, method, method_params)


    def dynamics(self, voltage_t0, threshold_t0, AScurrents_t0, inj, time_step, spike_time_steps):    
        """ Update the voltage, threshold, and afterspike currents of the neuron for a single time step.
        :parameter voltage_t0: the current voltage of the neuron
        :type voltage_t0: float
        :parameter threshold_t0: the current spike threshold level of the neuron
        :type threshold_t0: float
        :parameter AScurrents_t0: the current state of the afterspike currents in the neuron
        :type AScurrents_t0: np.ndarray
        :parameter inj: the current value of the current injection into the neuron
        :type inj: float
        :parameter time_step: the current time step of the neuron simulation
        :type time_step: int
        :parameter spike_time_steps: a list of all of the time steps of spikes in the neuron
        :type spike_time_steps: list
        :returns: voltage_t1 (voltage at next time step), threshold_t1 (threshold at next time step), AScurrents_t1 (afterspike currents at next time step)
        :rtype: tuple
        """

        AScurrents_t1 = self.AScurrent_dynamics_method(self, AScurrents_t0, time_step, spike_time_steps)
        voltage_t1 = self.voltage_dynamics_method(self, voltage_t0, AScurrents_t0, inj)
        threshold_t1 = self.threshold_dynamics_method(self, threshold_t0, voltage_t0)

        return voltage_t1, threshold_t1, AScurrents_t1
    
        
    def reset(self, voltage_t0, threshold_t0, AScurrents_t0):
        """ Apply reset rules to the neuron's voltage, threshold, and afterspike currents assuming a spike has occurred (voltage is above threshold). 

        :parameter voltage_t0: the current voltage of the neuron
        :type voltage_t0: float
        :parameter threshold_t0: the current spike threshold level of the neuron
        :type threshold_t0: float
        :parameter AScurrents_t0: the current state of the afterspike currents in the neuron
        :type AScurrents_t0: np.ndarray
        :returns: voltage_t1 (voltage at next time step), threshold_t1 (threshold at next time step), AScurrents_t1 (afterspike currents at next time step)
        :rtype: tuple
        """
        
        AScurrents_t1 = self.AScurrent_reset_method(self, AScurrents_t0)  
        voltage_t1 = self.voltage_reset_method(self, voltage_t0)
        threshold_t1 = self.threshold_reset_method(self, threshold_t0, voltage_t1)

        if voltage_t1 > threshold_t1:
            Exception("Voltage reset above threshold at time step (%f): voltage_t1 (%f) threshold_t1 (%f), voltage_t0 (%f) threshold_t0 (%f) AScurrents_t0 (%s)" % (t, voltage_t1, threshold_t1, voltage_t0, threshold_t0, repr(AScurrents_t0)))

        return voltage_t1, threshold_t1, AScurrents_t1
    
    def run(self, stim):
        """ Run neuron simulation over a given stimulus. This steps through the stimulus applying dynamics equations.
        After each step it checks if voltage is above threshold.  If so, self.spike_cut_length NaNs are inserted 
        into the output voltages, reset rules are applied to the voltage, threshold, and afterspike currents, and the 
        simulation resumes.

        :parameter stim: vector of scalar current values
        :type stim: np.ndarray
        :returns: a dictionary containing: 
          'voltage': simulated voltage values, 
          'threshold': threshold values during the simulation,
          'AScurrents': afterspike current values during the simulation, 
          'grid_spike_times': spike times (in uits of self.dt) aligned to simulation time steps, 
          'interpolated_spike_times': spike times (in units of self.dt) linearly interpolated between time steps, 
          'spike_time_steps': the indices of grid spike times, 
          'interpolated_spike_voltage': voltage of the simulation at interpolated spike times, 
          'interpolated_spike_threshold': threshold of the simulation at interpolated spike times
        :rtype: dict
        """

        # initialize the voltage, threshold, and afterspike current values
        voltage_t0 = self.init_voltage
        threshold_t0 = self.init_threshold
        AScurrents_t0 = self.init_AScurrents

        num_time_steps = len(stim) 
        num_AScurrents = len(AScurrents_t0)
        
        # pre-allocate the output voltages, thresholds, and after-spike currents
        voltage_out=np.empty(num_time_steps)
        voltage_out[:]=np.nan
        threshold_out=np.empty(num_time_steps)
        threshold_out[:]=np.nan
        AScurrents_out=np.empty(shape=(num_time_steps, num_AScurrents))
        AScurrents_out[:]=np.nan        

        # array that will hold spike indices
        spike_time_steps = []
        grid_spike_times = []
        interpolated_spike_times = []
        interpolated_spike_voltage = []
        interpolated_spike_threshold = []

        time_step = 0
        while time_step < num_time_steps:
            if time_step % 10000 == 0:
                logging.info("time step %d / %d" % (time_step,  num_time_steps))

            # compute voltage, threshold, and ascurrents at current time step
            (voltage_t1, threshold_t1, AScurrents_t1) = self.dynamics(voltage_t0, threshold_t0, AScurrents_t0, stim[time_step], time_step, spike_time_steps) 

            #if the voltage is bigger than the threshold record the spike and reset the values
            if voltage_t1 > threshold_t1: 

                # spike_time_steps are stimulus indices when voltage surpassed threshold
                spike_time_steps.append(time_step)
                grid_spike_times.append(time_step * self.dt) 

                # compute higher fidelity spike time/voltage/threshold by linearly interpolating 
                interpolated_spike_times.append(interpolate_spike_time(self.dt, time_step, threshold_t0, threshold_t1, voltage_t0, voltage_t1))

                interpolated_spike_time_offset = interpolated_spike_times[-1] - (time_step - 1) * self.dt
                interpolated_spike_voltage.append(interpolate_spike_value(self.dt, interpolated_spike_time_offset, voltage_t0, voltage_t1))
                interpolated_spike_threshold.append(interpolate_spike_value(self.dt, interpolated_spike_time_offset, threshold_t0, threshold_t1))
            
                # reset voltage, threshold, and afterspike currents
                (voltage_t0, threshold_t0, AScurrents_t0) = self.reset(voltage_t1, threshold_t1, AScurrents_t1) 

                # if we are not integrating during the spike (which includes right now), insert nans then jump ahead
                if self.spike_cut_length > 0:
                    n = self.spike_cut_length
                    voltage_out[time_step:time_step+n] = np.nan
                    threshold_out[time_step:time_step+n] = np.nan
                    AScurrents_out[time_step:time_step+n,:] = np.nan

                    time_step += self.spike_cut_length
                else:
                    # we are integrating during the spike, so store the reset values
                    voltage_out[time_step] = voltage_t0 
                    threshold_out[time_step] = threshold_t0
                    AScurrents_out[time_step,:] = AScurrents_t0

                    time_step += 1
            else:
                # there was no spike, store the next voltages
                voltage_out[time_step] = voltage_t1 
                threshold_out[time_step] = threshold_t1
                AScurrents_out[time_step,:] = AScurrents_t1

                voltage_t0 = voltage_t1
                threshold_t0 = threshold_t1
                AScurrents_t0 = AScurrents_t1
                
                time_step += 1

        return {
            'voltage': voltage_out, 
            'threshold': threshold_out, 
            'AScurrents': AScurrents_out,
            'grid_spike_times': np.array(grid_spike_times), 
            'interpolated_spike_times': np.array(interpolated_spike_times), 
            'spike_time_steps': np.array(spike_time_steps), 
            'interpolated_spike_voltage': np.array(interpolated_spike_voltage), 
            'interpolated_spike_threshold': np.array(interpolated_spike_threshold)
            }


    def run_with_biological_spikes(self, stim, bio_spike_time_steps):
        """ Run the neuron simulation over a stimulus, but do not allow the model to spike on its own.  Rather,
        force the simulation to spike and reset at a given set of spike indices.  Dynamics rules are applied
        between spikes regardless of the simulated voltage and threshold values.  Reset rules are applied only 
        at input spike times. This is used during optimization to force the model to follow the spikes of biological data.
        The model is optimized in this way so that history effects due to spiking can be adequately modeled.  For example, 
        every time the model spikes a new set of afterspike currents will be initiated. To ensure that afterspike currents 
        can be optimized, we force them to be initiated at the time of the biological spike.

        :parameter stim: vector of scalar current values
        :type stim: np.ndarray
        :parameter bio_spike_time_steps: spike time step indices
        :returns: a dictionary containing:
          'voltage': simulated voltage values,
          'threshold': simulated threshold values,
          'AScurrent_matrix': afterspike currents during the simulation,
          'grid_model_spike_times': spike times of the model aligned to the simulation grid (when it would have spiked),
          'interpolated_model_spike_times': spike times of the model linearly interpolated between time steps,
          'grid_ISI': interspike interval between grid model spike times,
          'interpolated_ISI': interspike interval between interpolated model spike times,
          'grid_bio_spike_model_voltage': voltage of the model at biological/input spike times,
          'grid_bio_spike_model_threshold': voltage of the model at biological/input spike times interpolated between time steps
        :rtype: dict
        """

        voltage_t0 = self.init_voltage
        threshold_t0 = self.init_threshold
        AScurrents_t0 = self.init_AScurrents

        start_index = 0
        end_index = 0

        try:
            num_spikes = len(bio_spike_time_steps)
            
            # if there are no target spikes, just run until the model spikes
            if num_spikes == 0:  

                start_index = 0
                end_index = len(stim)

                # evaluate the model starting from the beginning until the model spikes
                run_data = self.run_until_biological_spike(voltage_t0, threshold_t0, AScurrents_t0, 
                                                           stim, start_index, end_index, 
                                                           []) 

                voltage = run_data['voltage']
                threshold = run_data['threshold']
                AScurrent_matrix = run_data['AScurrent_matrix']

                if len(voltage) != len(stim):
                    warnings.warn('Your voltage output is not the same length as your stimulus')
                if len(threshold) != len(stim):
                    warnings.warn('Your threshold output is not the same length as your stimulus')
                if len(AScurrent_matrix) != len(stim):
                    warnings.warn('Your AScurrent_matrix output is not the same length as your stimulus')
                                          
                # do not keep track of the spikes in the model that spike if the target doesn't spike.
                grid_ISI = np.array([])
                interpolated_ISI = np.array([])

                grid_model_spike_times = np.array([]) 
                interpolated_model_spike_times = np.array([]) 

                grid_bio_spike_model_voltage = np.array([])
                grid_bio_spike_model_threshold = np.array([])
            else:
                # initialize the output arrays
                grid_ISI = np.empty(num_spikes)
                interpolated_ISI = np.empty(num_spikes)

                grid_model_spike_times = np.empty(num_spikes)
                interpolated_model_spike_times = np.empty(num_spikes)

                grid_bio_spike_model_voltage = np.empty(num_spikes)
                grid_bio_spike_model_threshold = np.empty(num_spikes)

                spikeIndStart = 0
                
                voltage = np.empty(len(stim))
                voltage[:] = np.nan  
                threshold = np.empty(len(stim))
                threshold[:] = np.nan
                AScurrent_matrix = np.empty(shape=(len(stim), len(AScurrents_t0)))
                AScurrent_matrix[:] = np.nan
        
                # run the simulation over the interspike intervals (starting at the beginning of the simulation). 
                start_index = 0
                for spike_num in range(num_spikes):

                    if spike_num % 10 == 0:
                        logging.debug("spike %d / %d" % (spike_num,  num_spikes))

                    end_index = int(bio_spike_time_steps[spike_num])
    
                    assert start_index < end_index, Exception("start_index > end_index: this is probably because spike_cut_length is longer than the previous inter-spike interval")

                    # run the simulation over this interspike interval
                    run_data = self.run_until_biological_spike(voltage_t0, threshold_t0, AScurrents_t0, 
                                                               stim, start_index, end_index, 
                                                               bio_spike_time_steps)

                    # assign the simulated data to the correct locations in the output arrays
                    voltage[start_index:end_index] = run_data['voltage']
                    threshold[start_index:end_index] = run_data['threshold']
                    AScurrent_matrix[start_index:end_index,:] = run_data['AScurrent_matrix']

                    grid_ISI[spike_num] = run_data['grid_model_spike_time']
                    interpolated_ISI[spike_num] = run_data['interpolated_model_spike_time']

                    grid_model_spike_times[spike_num] = run_data['grid_model_spike_time'] + start_index * self.dt 
                    interpolated_model_spike_times[spike_num] = run_data['interpolated_model_spike_time'] + start_index * self.dt 

                    grid_bio_spike_model_voltage[spike_num] = run_data['grid_bio_spike_model_voltage']
                    grid_bio_spike_model_threshold[spike_num] = run_data['grid_bio_spike_model_threshold']

                    # update the voltage, threshold, and afterspike currents for the next interval
                    voltage_t0 = run_data['voltage_t0']
                    threshold_t0 = run_data['threshold_t0']
                    AScurrents_t0 = run_data['AScurrents_t0']
                    
                    start_index = end_index

                    # if cutting spikes, jump forward the appropriate amount of time
                    if self.spike_cut_length > 0:
                        start_index += self.spike_cut_length
                
                # simulate the portion of the stimulus between the last spike and the end of the array.
                # no spikes are recorded from this time!
                run_data = self.run_until_biological_spike(voltage_t0, threshold_t0, AScurrents_t0, 
                                                           stim, start_index, len(stim),
                                                           bio_spike_time_steps) 

                voltage[start_index:] = run_data['voltage']
                threshold[start_index:] = run_data['threshold']
                AScurrent_matrix[start_index:,:] = run_data['AScurrent_matrix']

            # make sure that the output data has the correct number of spikes in it
            if ( len(interpolated_model_spike_times) != num_spikes or 
                 len(grid_model_spike_times) != num_spikes or 
                 len(grid_ISI) != num_spikes or 
                 len(interpolated_ISI) != num_spikes or 
                 len(grid_bio_spike_model_voltage) != num_spikes or 
                 len(grid_bio_spike_model_threshold) != num_spikes):
                raise Exception('The number of spikes in your output does not match your target')

        except GLIFNeuronException, e:
            
            # if an exception was raised during run_until_spike, record any simulated data before exiting
            voltage[start_index:end_index] = e.data['voltage']
            threshold[start_index:end_index] = e.data['threshold']
            AScurrent_matrix[start_index:end_index,:] = e.data['AScurrent_matrix']

            out = {
                'voltage': voltage,
                'threshold': threshold,
                'AScurrent_matrix': AScurrent_matrix,

                'grid_ISI': grid_ISI,
                'interpolated_ISI': interpolated_ISI,

                'grid_model_spike_times': grid_model_spike_times,
                'interpolated_model_spike_times': interpolated_model_spike_times,

                'grid_bio_spike_model_voltage': grid_bio_spike_model_voltage,
                'grid_bio_spike_model_threshold': grid_bio_spike_model_threshold
                }

            raise GLIFNeuronException(e.message, out)

        return {
            'voltage': voltage,
            'threshold': threshold,
            'AScurrent_matrix': AScurrent_matrix,

            'grid_model_spike_times': grid_model_spike_times,
            'interpolated_model_spike_times': interpolated_model_spike_times,

            'grid_ISI': grid_ISI,
            'interpolated_ISI': interpolated_ISI,

            'grid_bio_spike_model_voltage': grid_bio_spike_model_voltage,
            'grid_bio_spike_model_threshold': grid_bio_spike_model_threshold
        }
                
    def run_until_biological_spike(self, voltage_t0, threshold_t0, AScurrents_t0, 
                                   stim, start_index, end_index, 
                                   bio_spike_time_steps):
        """ Run the neuron simulation over a segment of a stimulus given initial conditions. The model simulates
        until either the model spikes or the end of the segment is reached.  If the model does not spike, a 
        spike time is extrapolated past the end of the simulation segment.

        :parameter voltage_t0: the current voltage of the neuron
        :type voltage_t0: float
        :parameter threshold_t0: the current spike threshold level of the neuron
        :type threshold_t0: float
        :parameter AScurrents_t0: the current state of the afterspike currents in the neuron
        :type AScurrents_t0: np.ndarray
        :parameter stim: the full stimulus array (not just the segment of data being simulated)
        :type stim: np.ndarray
        :parameter start_index: index to start simulating
        :type start_index: int
        :parameter end_index: index *after* the last index to be simulated
        :type end_index: int
        :parameter bio_spike_time_steps: time steps of input spikes
        :type bio_spike_time_steps: list
        :returns: a dictionary containing:
            'voltage': simulated voltage value
            'threshold': simulated threshold values
            'AScurrent_matrix': afterspike current values during the simulation
            'grid_model_spike_time': model spike time (in units of dt) 
            'interpolated_model_spike_time': model spike time (in units of dt) interpolated between time steps
            'voltage_t0': reset voltage value to be used in subsequent simulation interval 
            'threshold_t0': reset threshold value to be used in subsequent simulation interval
            'AScurrents_t0': reset afterspike current value to be used in subsequent simulation interval
            'grid_bio_spike_model_voltage': model voltage at the time of the input spike
            'grid_bio_spike_model_threshold': model threshold at the time of the input spike
        :rtype: dict
        """

        # preallocate arrays and matricies
        num_time_steps = end_index - start_index
        num_spikes = len(bio_spike_time_steps)

        voltage_out = np.empty(num_time_steps)
        voltage_out[:] = np.nan
        threshold_out = np.empty(num_time_steps)
        threshold_out[:] = np.nan
        AScurrent_matrix = np.empty(shape=(num_time_steps, len(AScurrents_t0)))
        AScurrent_matrix[:] = np.nan

        grid_model_spike_time = None
        interpolated_model_spike_time = None
        
        # calculate the model values between the two target spikes (don't stop if there is a spike)
        for time_step in xrange(num_time_steps):

            # Note that here you are not recording the first v0 because that was recoded at the end of the previous spike
            voltage_out[time_step] = voltage_t0 
            threshold_out[time_step] = threshold_t0
            AScurrent_matrix[time_step,:] = np.matrix(AScurrents_t0) 
            
            if np.isnan(voltage_t0) or np.isinf(voltage_t0) or np.isnan(threshold_t0) or np.isinf(threshold_t0) or any(np.isnan(AScurrents_t0)) or any(np.isinf(AScurrents_t0)):
                logging.error(self)
                logging.error('time step: %d / %d' % (time_step, num_time_steps))
                logging.error('    voltage_t0: %f' % voltage_t0)
                logging.error('    voltage started the run at: %f' % voltage_out[0])
                logging.error('    voltage before: %s' % voltage_out[time_step-20:time_step])
                logging.error('    threshold_t0: %f' % threshold_t0)
                logging.error('    threshold started the run at: %f' % threshold_out[0])
                logging.error('    threshold before: %s' % threshold_out[time_step-20:time_step])
                logging.error('    AScurrents_t0: %s' % AScurrents_t0)
                raise GLIFNeuronException('Invalid threshold, voltage, or after-spike current encountered.', {
                        'voltage': voltage_out,
                        'threshold': threshold_out,
                        'AScurrent_matrix': AScurrent_matrix
                        })
            
            (voltage_t1, threshold_t1, AScurrents_t1) = self.dynamics(voltage_t0, threshold_t0, AScurrents_t0, stim[time_step+start_index], time_step+start_index, bio_spike_time_steps) #TODO fix list versus array
            
            voltage_t0=voltage_t1
            threshold_t0=threshold_t1
            AScurrents_t0=AScurrents_t1
            
        # figure out whether model neuron spiked or not        
        for time_step in range(0, num_time_steps): 
            if voltage_out[time_step] > threshold_out[time_step]:
                grid_model_spike_time = self.dt * time_step #not that this should be time_step even though it is index+1 in runModel function because here it isnt recorded until the next step
                interpolated_model_spike_time = interpolate_spike_time(self.dt, time_step-1, threshold_out[time_step-1], threshold_out[time_step], voltage_out[time_step-1], voltage_out[time_step])
                break
                
        # if the last voltage is above threshold and there hasn't already been a spike
        if voltage_t1 > threshold_t1 and grid_model_spike_time is None: 
            grid_model_spike_time = self.dt*num_time_steps
            interpolated_model_spike_time = interpolate_spike_time(self.dt, num_time_steps - 1, threshold_out[num_time_steps-1], threshold_t1, voltage_out[num_time_steps-1], voltage_t1)

        #if the model never spiked, extrapolate to guess when it would have spiked
        if grid_model_spike_time is None: 
            interpolated_model_spike_time = extrapolate_spike_time(self.dt, num_time_steps, threshold_out[-1], threshold_t1, voltage_out[-1], voltage_t1)
            grid_model_spike_time = np.ceil(interpolated_model_spike_time / self.dt) * self.dt  #grid spike time based off extrapolated spike time
        
        # if the target spiked, reset so that next round will start at reset but not recording it in the voltage here.
        # note that at the last section of the stimulus where there is no current injected the model will be reset even if
        # the biological neuron doesn't spike.  However, this doesnt matter as it won't be recorded. 
        if num_spikes > 0:
            (voltage_t0, threshold_t0, AScurrents_t0) = self.reset(voltage_t1, threshold_t1, AScurrents_t1)
        
        return {
            'voltage': voltage_out, 
            'threshold': threshold_out, 
            'AScurrent_matrix': AScurrent_matrix, 

            'grid_model_spike_time': grid_model_spike_time,
            'interpolated_model_spike_time': interpolated_model_spike_time, 

            'voltage_t0': voltage_t0, 
            'threshold_t0': threshold_t0, 
            'AScurrents_t0': AScurrents_t0, 

            'grid_bio_spike_model_voltage': voltage_t1, 
            'grid_bio_spike_model_threshold': threshold_t1            
            }



def extrapolate_spike_time(dt, num_time_steps, threshold_t0, threshold_t1, voltage_t0, voltage_t1):
    """ Given two voltage and threshold values and an interval between them, extrapolate a spike time
    by intersecting lines the thresholds and voltages. """
    return line_crossing_x(dt * num_time_steps, voltage_t0, voltage_t1, threshold_t0, threshold_t1)

def interpolate_spike_time(dt, time_step, threshold_t0, threshold_t1, voltage_t0, voltage_t1):
    """ Given two voltage and threshold values, the dt between them and the initial time step, interpolate
    a spike time within the dt interval by intersecting the two lines. """
    return time_step*dt + line_crossing_x(dt, voltage_t0, voltage_t1, threshold_t0, threshold_t1)

def interpolate_spike_value(dt, interpolated_spike_time_offset, v0, v1):
    """ Take a value at two adjacent time steps and linearly interpolate what the value would be
    at an offset between the two time steps. """
    return v0 + (v1 - v0) * interpolated_spike_time_offset / dt

def line_crossing_x(dx, a0, a1, b0, b1):
    """ Find the x value of the intersection of two lines. """
    return dx * (b0 - a0) / ( (a1 - a0) - (b1 - b0) )
