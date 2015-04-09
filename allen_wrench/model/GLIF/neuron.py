import logging

import numpy as np
import json, utilities
import copy

from neuron_methods import GLIFNeuronMethod, METHOD_LIBRARY

class GLIFBadResetException( Exception ):
    """ Exception raised when voltage is still above threshold after a reset rule is applied. """
    def __init__(self, message, dv):
        super(Exception, self).__init__(message)
        self.dv = dv
            
class GLIFNeuron( object ):    
    """ Implements the current-based Mihalas Neiber GLIF neuron.  Simulations model the voltage, 
    threshold, and afterspike currents of a neuron given an input stimulus.  A set of modular dynamics
    rules are applied until voltage crosses threshold, at which point a set of modular reset rules are 
    applied. See neuron_methods.py for a list of what options there are for voltage, threshold, and
    afterspike current dynamics and reset rules.

    Parameters
    ----------
     El : float 
         resting potential 
     dt : float
         duration between time steps
     tau: np.ndarray
         TODO
     R_input : float
         input resistance
     C : float
         capacitance
     asc_vector : np.ndarray
         afterspike current vector.  one element per element of tau.
     spike_cut_length : int
         how many time steps to replace with NaNs when a spike occurs.
     th_inf : float
         instantaneous threshold
     coeffs : doct
        dictionary coefficients premultiplied to neuron properties during simulation. used for optimization.
     AScurrent_dynamics_method : dict
         dictionary containing the 'name' of the afterspike current dynamics method to use and a 'params' dictionary parameters to pass to that function.
     voltage_dynamics_method : dict
         dictionary containing the 'name' of the voltage dynamics method to use and a 'params' dictionary parameters to pass to that function.
     threshold_dynamics_method : dict
         dictionary containing the 'name' of the threshold dynamics method to use and a 'params' dictionary parameters to pass to that function.
     AScurrent_reset_method : dict
         dictionary containing the 'name' of the afterspike current dynamics method to use and a 'params' dictionary parameters to pass to that function.
     voltage_reset_method : dict
         dictionary containing the 'name' of the voltage dynamics method to use and a 'params' dictionary parameters to pass to that function.
     threshold_reset_method : dict
         dictionary containing the 'name' of the threshold dynamics method to use and a 'params' dictionary parameters to pass to that function.
     init_voltage : float 
        initial voltage value
     init_threshold : float
         initial spike threshold value
     init_AScurrents : np.ndarray
        initial afterspike current vector. one element per element of tau.
    """

    TYPE = "GLIF"

    def __init__(self, El, dt, tau, R_input, C, asc_vector, spike_cut_length, th_inf, th_adapt, coeffs,
                 AScurrent_dynamics_method, voltage_dynamics_method, threshold_dynamics_method,
                 AScurrent_reset_method, voltage_reset_method, threshold_reset_method,
                 init_method_data, init_voltage, init_threshold, init_AScurrents): 

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
        self.th_adapt=th_adapt

        self.init_method_data = init_method_data
        self.update_method_data = copy.deepcopy(init_method_data)

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

    @classmethod
    def from_dict(cls, d):
        return cls(El = d['El'],
                   dt = d['dt'],
                   tau = d['tau'],
                   R_input = d['R_input'],
                   C = d['C'],
                   asc_vector = d['asc_vector'],
                   spike_cut_length = d['spike_cut_length'],
                   th_inf = d['th_inf'],
                   th_adapt=d['th_adapt'],
                   coeffs = d.get('coeffs', {}),
                   AScurrent_dynamics_method = d['AScurrent_dynamics_method'],
                   voltage_dynamics_method = d['voltage_dynamics_method'],
                   threshold_dynamics_method = d['threshold_dynamics_method'],
                   voltage_reset_method = d['voltage_reset_method'],
                   AScurrent_reset_method = d['AScurrent_reset_method'],
                   threshold_reset_method = d['threshold_reset_method'],
                   init_method_data = d.get('init_method_data', {}),
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
            'init_method_data': self.init_method_data,
            'update_method_data': self.update_method_data,
            'init_voltage': self.init_voltage,
            'init_threshold': self.init_threshold,
            'init_AScurrents': self.init_AScurrents
        }

    @staticmethod
    def configure_method(method_name, method, method_params):
        """ Create a GLIFNeuronMethod instance given a name, a function, and function parameters. 
        This is just a shortcut to the GLIFNeuronMethod constructor.

        Parameters
        ----------
        method_name : string
            name for refering to this method later
        method : function
            a python function 
        method_parameters : dict
           function arguments whose values should be fixed

        Returns
        -------
        GLIFNeuronMethod
            a GLIFNeuronMethod instance
        """ 

        return GLIFNeuronMethod(method_name, method, method_params)

    @staticmethod
    def configure_library_method(method_type, params):
        """ Create a GLIFNeuronMethod instance out of a library of functions organized by type name. 
        This refers to the METHOD_LIBRARY in neuron_methods.py, which lays out the available functions 
        that can be used for dynamics and reset rules.

        Parameters
        ----------
        method_type : string
            the name of a function category (e.g. 'AScurrent_dynamics_method' for the afterspike current dynamics methods)
        params : dict
            a dictionary with two members. 'name': the string name of function you want, and 'params': parameters you want to pass to that function

        Returns
        -------
        GLIFNeuronMethod
            a GLIFNeuronMethod instance
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

    def reset_method_data(self):
        self.update_method_data = copy.deepcopy(self.init_method_data)

    def dynamics(self, voltage_t0, threshold_t0, AScurrents_t0, inj, time_step, spike_time_steps):    
        """ Update the voltage, threshold, and afterspike currents of the neuron for a single time step.

        Parameters
        ----------
        voltage_t0 : float
            the current voltage of the neuron
        threshold_t0 : float
            the current spike threshold level of the neuron
        AScurrents_t0 : np.ndarray
            the current state of the afterspike currents in the neuron
        inj : float
            the current value of the current injection into the neuron
        time_step : int
            the current time step of the neuron simulation
        spike_time_steps : list
            a list of all of the time steps of spikes in the neuron

        Returns
        -------
        tuple
            voltage_t1 (voltage at next time step), threshold_t1 (threshold at next time step), AScurrents_t1 (afterspike currents at next time step)
        """

        AScurrents_t1 = self.AScurrent_dynamics_method(self, AScurrents_t0, time_step, spike_time_steps)
        voltage_t1 = self.voltage_dynamics_method(self, voltage_t0, AScurrents_t0, inj)
        threshold_t1 = self.threshold_dynamics_method(self, threshold_t0, voltage_t0)

        return voltage_t1, threshold_t1, AScurrents_t1
    
        
    def reset(self, voltage_t0, threshold_t0, AScurrents_t0):
        """ Apply reset rules to the neuron's voltage, threshold, and afterspike currents assuming a spike has occurred (voltage is above threshold). 

        Parameters
        ----------
        voltage_t0 : float
           the current voltage of the neuron
        threshold_t0 : float
            the current spike threshold level of the neuron
        AScurrents_t0 : np.ndarray
            the current state of the afterspike currents in the neuron

        Returns
        -------
        tuple
            voltage_t1 (voltage at next time step), threshold_t1 (threshold at next time step), AScurrents_t1 (afterspike currents at next time step)
        """
        
        AScurrents_t1 = self.AScurrent_reset_method(self, AScurrents_t0)  
        voltage_t1 = self.voltage_reset_method(self, voltage_t0)
        threshold_t1 = self.threshold_reset_method(self, threshold_t0, voltage_t1)

        if voltage_t1 > threshold_t1:
            raise GLIFBadResetException("Voltage reset above threshold: voltage_t1 (%f) threshold_t1 (%f), voltage_t0 (%f) threshold_t0 (%f) AScurrents_t0 (%s)" % ( voltage_t1, threshold_t1, voltage_t0, threshold_t0, repr(AScurrents_t0)), voltage_t1 - threshold_t1)

        return voltage_t1, threshold_t1, AScurrents_t1
    
    def run(self, stim):
        """ Run neuron simulation over a given stimulus. This steps through the stimulus applying dynamics equations.
        After each step it checks if voltage is above threshold.  If so, self.spike_cut_length NaNs are inserted 
        into the output voltages, reset rules are applied to the voltage, threshold, and afterspike currents, and the 
        simulation resumes.

        Parameters
        ----------
        stim : np.ndarray
            vector of scalar current values

        Returns
        -------
        dict
            a dictionary containing: 
                'voltage': simulated voltage values, 
                'threshold': threshold values during the simulation,
                'AScurrents': afterspike current values during the simulation, 
                'grid_spike_times': spike times (in uits of self.dt) aligned to simulation time steps, 
                'interpolated_spike_times': spike times (in units of self.dt) linearly interpolated between time steps, 
                'spike_time_steps': the indices of grid spike times, 
                'interpolated_spike_voltage': voltage of the simulation at interpolated spike times, 
                'interpolated_spike_threshold': threshold of the simulation at interpolated spike times
        """

        # initialize the voltage, threshold, and afterspike current values
        voltage_t0 = self.init_voltage
        threshold_t0 = self.init_threshold
        AScurrents_t0 = self.init_AScurrents

        self.reset_method_data() #get rid of lingering method data

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
    assert type(a0) != int and type(a1) != int and type(b0) != int and type(b1) != int, Exception("Do not pass integers into this function!")
    return dx * (b0 - a0) / ( (a1 - a0) - (b1 - b0) )
        

def line_crossing_y(dx, a0, a1, b0, b1):
    """ Find the y value of the intersection of two lines. """
    return b0 + (b1 - b0) * (b0 - a0) / ((a1 - a0) - (b1 - b0))

