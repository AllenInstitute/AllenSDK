# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

""" The methods in this module are used for configuring dynamics and reset rules for the GlifNeuron.  
For more details on how to use these methods, see :doc:`glif_models`.
"""

import functools
import numpy as np


class GlifNeuronMethod( object ):
    """ A simple class to keep track of the name and parameters associated with a neuron method.
    This class is initialized with a name, function, and parameters to pass to the function.  The
    function then has those passed parameters fixed to a partial function using functools.partial.  
    This class then mimics a function itself using the __call__ convention. Parameters that are not 
    fixed in this way are assumed to be passed into the method when it is called.  If the passed
    parameters contain an argument that is not part of the function signature, an exception will
    be raised.    

    Parameters
    ----------
    method_name : string
        A shorthand name that will be used to reference this method in the `GlifNeuron`.
    method : function
        A python function to be called when this instance is called.
    method_params : dict
        A dictionary mapping function arguments to values for values that should be fixed.
    """

    def __init__(self, method_name, method, method_params):
        self.name = method_name
        self.params = method_params
        self.method = functools.partial(method, **method_params)

    def __call__(self, *args, **kwargs):
        """ Defining this method allows an instance to be called like a function """
        return self.method(*args, **kwargs)

    def to_dict(self):
        return {
            'name': self.name,
            'params': self.params
        }

    def modify_parameter(self, param, operator):
        """ Modify a function parameter that needs to be modified after initialization. 

        Parameters
        ----------
        param : string
            the name of the parameter to modify
        operator : callable
            a function or lambda that returns the desired modified value

        Returns
        -------
        type
            the new value of the variable that was just modified.
        """
        value = operator(self.method.keywords[param])
        self.method.keywords[param] = value
        return value


def dynamics_AScurrent_exp(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ Exponential afterspike current dynamics method takes a current at t0 and returns the current at
    a time step later.
    
    Parameters
    ----------
    AScurrents_t0: np.ndarray
        value of the after-spike currents 
    time_step: int
        index of global time step
    spike_time_steps: list
        index of global time of all model spikes

    Returns
    -------
    type: np.ndarray
        the new value of after-spike currents.

    """
    return AScurrents_t0*(1.0 + neuron.k*neuron.dt)
    

def dynamics_AScurrent_none(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ This method always returns zeros for the afterspike currents, regardless of input. 
    
    Parameters
    ----------
    AScurrents_t0: np.ndarray
        value of the after-spike currents 
    time_step: int
        index of global time step
    spike_time_steps: list
        index of global time of all model spikes


    Returns
    -------
    type: np.ndarray
        array of zeros for the after-spike currents.

    """
    return np.zeros(len(AScurrents_t0))


def dynamics_voltage_forward_euler(neuron, voltage_t0, AScurrents_t0, inj):
    """ Linear voltage dynamics simulated by standard forward Euler. 
    
    Parameters
    ----------
    voltage_t0: float
        value of voltage
    AScurrents_t0: np.ndarray
        value of the after-spike currents 
    inj: float
        value of stimulus current being injected

    Returns
    -------
    type: float
        value of voltage at the next time step.

    """
    return voltage_t0 + (inj + np.sum(AScurrents_t0) - neuron.G * neuron.coeffs['G'] * (voltage_t0 - neuron.El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])

def dynamics_voltage_euler_exact(neuron, voltage_t0, AScurrents_t0, inj):
    """ Linear voltage dynamics simulated by the exponential kernel to calculate the exact voltage values at the time step.
        This assumes constant current (calculated as an average) over the time step. 
    
    Parameters
    ----------
    voltage_t0: float
        value of voltage
    AScurrents_t0: np.ndarray
        value of the after-spike currents 
    inj: float
        value of stimulus current being injected

    Returns
    -------
    type: float
        value of voltage at the next time step

    """

    tau = (neuron.C * neuron.coeffs['C'])
    I = inj + np.sum(AScurrents_t0)
    g = neuron.G * neuron.coeffs['G']
    A = g/tau
    N = (I+ g*neuron.El)/tau

    return voltage_t0*np.exp(-neuron.dt*A) + N*(1-np.exp(-A*neuron.dt))/A
    

def dynamics_threshold_three_sep_components(neuron, threshold_t0, voltage_t0):        
    """The threshold will adapt via two mechanisms: 1. a slower voltage dependent adaptation as in the dynamics_threshold_adapt_standard. 
    These are the components which are fit via optimization and inditial conditions are supplied via the GLM. 2. A fast component initiated 
    by a spike which quickly decays.  These values are estimated via the tri short square stimuli. 
    
    Parameters
    ----------
    threshold_t0: float
        value of threshold
    voltage_t0: float
        value of voltage

    Returns
    -------
    type: float
        value of overall threshold at the next time step.

    """
    
    #update_method_data keeps track of parameters that need to be shared between methods
    md = neuron.update_method_data

    # initial conditions
    if 'th_spike' not in md:
        md['th_spike'] = [ 0 ]
    if 'th_voltage' not in md:
        md['th_voltage'] = [ 0 ]

    th_spike = md['th_spike'][-1]
    th_voltage = md['th_voltage'][-1] 

    voltage_component = th_voltage + ( md['a_voltage'] *  ( voltage_t0 - neuron.El ) - 
                                       md['b_voltage'] * neuron.coeffs['b'] * ( th_voltage ) ) * neuron.dt
    spike_component = th_spike - md['b_spike'] * th_spike * neuron.dt

    #------update the subtrheshold voltage and spiking values of the threshold
    md['th_spike'].append(spike_component)
    md['th_voltage'].append(voltage_component)
    
    return neuron.coeffs['a']*voltage_component+spike_component+neuron.th_inf * neuron.coeffs['th_inf']    
    
def dynamics_threshold_inf(neuron, threshold_t0, voltage_t0):
    """ Set threshold to the neuron's instantaneous threshold.
    
    Parameters
    ----------
    threshold_t0: float
        value of threshold
    voltage_t0: float
        value of voltage

    Returns
    -------
    type: float
        value of overall threshold at the next time step.
    """
    return neuron.coeffs['th_inf'] * neuron.th_inf
    

def reset_AScurrent_sum(neuron, AScurrents_t0, r):
    """ Reset afterspike currents by adding summed exponentials. Left over currents from last spikes as 
    well as newly initiated currents from current spike.  Currents amplitudes in neuron.asc_vector need
    to be the amplitudes advanced though the spike cutting.  I.e. In the preprocessor if the after spike currents 
    are calculated via the GLM from spike initiation the amplitude at the time after the spike cutting needs to 
    be calculated and neuron.asc_vector needs to be set to this value.
    
    Parameters
    ----------
    r : np.ndarray
        a coefficient vector applied to the afterspike currents
    AScurrents_t0: np.ndarray
        value of the after-spike currents 

    Returns
    -------
    type: np.ndarray
        array of new after-spike currents.
    """
    new_currents=neuron.asc_vector * neuron.coeffs['asc_vector'] #neuron.asc_vector are amplitudes initiating after the spike is cut
    left_over_currents=AScurrents_t0 * r * np.exp(neuron.k * neuron.dt * neuron.spike_cut_length) #advancing cut currents though the spike    

    return new_currents+left_over_currents


def reset_AScurrent_none(neuron, AScurrents_t0):
    """ Reset afterspike currents to zero. 
    Parameters
    ----------
    AScurrents_t0: np.ndarray
        value of the after-spike currents 

    Returns
    -------
    type: np.ndarray
        array of zeros for the after-spike currents.

    """   
    if np.sum(AScurrents_t0)!=0:
        raise Exception('You are running a LIF but the AScurrents are not zero!')
    return np.zeros(len(AScurrents_t0))


def reset_voltage_bio_rules(neuron, voltage_t0, a, b):
    """ Reset voltage to the previous value with a scale and offset applied derived from the electrophysiological data.

    Parameters
    ----------
    voltage_t0: float
        value of voltage at spike
    a : float
        voltage scale constant
    b : float
        voltage offset constant

    Returns
    -------
    type: float
        value of voltage after the spike
    """

    return a*(voltage_t0)+b


def reset_voltage_zero(neuron, voltage_t0):
    """ Reset voltage to zero. 
    
    Parameters
    ----------
    voltage_t0: float
        value of voltage at spike

    Returns
    -------
    type: float
        zero value of voltage after the spike
    """

    return 0.0


def reset_threshold_inf(neuron, threshold_t0, voltage_v1):
    """ Reset the threshold to instantaneous threshold. 
    
    Parameters
    ----------
    threshold_t0: float
        value of threshold at spike
    voltage_v1: float
        voltage after reset

    Returns
    -------
    type: float
        value of threshold after the spike
    """
    return neuron.coeffs['th_inf'] * neuron.th_inf


def reset_threshold_for_three_sep_components(neuron, threshold_t0, voltage_v1):
    """This method resets the two components of the threshold: a spike
    component and a subthreshold voltage component which are added to the instantaneous threshold.
    
    Parameters
    ----------
    threshold_t0: float
        value of threshold at spike
    voltage_v1: float
        voltage after reset

    Returns
    -------
    type: float
        value of threshold after the spike
    """
    
    #update_method_data keeps track of parameters that need to be shared between methods
    md = neuron.update_method_data
    
    #------update the subtrheshold voltage and spiking values of the threshold
    md['th_spike'].append( md['th_spike'][-1] + md['a_spike'] )
    md['th_voltage'].append( md['th_voltage'][-1] ) #note these are the same value.

    return md['th_spike'][-1] + md['th_voltage'][-1] + neuron.th_inf * neuron.coeffs['th_inf']


#: The METHOD_LIBRARY constant groups dynamics and reset methods by group name (e.g. 'voltage_dynamics_method').  
#Those groups assign each method in this file a string name.  This is used by the GlifNeuron when initializing its dynamics and reset methods.
METHOD_LIBRARY = {
    'AScurrent_dynamics_method': { 
        'exp_glm': dynamics_AScurrent_exp,
        'none': dynamics_AScurrent_none
        },
    'voltage_dynamics_method': { 
        #Note that both of these functions simulate linear voltage dynamics. 
        #the "linear" uses first order (i.e. forward) Euler for simulation while the "euler_exact" 
        #uses the exponential kernel to calculate the exact voltage values at the time step.
        # This assumes constant current (calculated as an average) over the time step. 
        'linear': dynamics_voltage_forward_euler,
        'euler_exact':dynamics_voltage_euler_exact
        },
    'threshold_dynamics_method': {
        'adapt_sum_slow_fast':  dynamics_threshold_three_sep_components,                        
        'adapt_rebound': dynamics_threshold_three_sep_components,
        'inf': dynamics_threshold_inf
        },
    'AScurrent_reset_method': {
        'sum': reset_AScurrent_sum,
        'none': reset_AScurrent_none
        }, 
    'voltage_reset_method': {
        'v_before': reset_voltage_bio_rules,
        'zero': reset_voltage_zero,
        }, 
    'threshold_reset_method': {
        'inf': reset_threshold_inf,
        'adapt_sum_slow_fast': reset_threshold_for_three_sep_components
        }
}
