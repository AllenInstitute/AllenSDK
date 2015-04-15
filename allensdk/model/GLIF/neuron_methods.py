# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

""" The methods in this module are used for configuring dynamics and reset rules for the GLIFNeuron.  
For more details on how to use these methods, see :doc:`glif_models`.
"""

import functools
import numpy as np

class GLIFNeuronMethod( object ):
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
        A shorthand name that will be used to reference this method in the `GLIFNeuron`.
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
        """ Modify a function parameter needs to be modified after initialization. 

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


def two_lines(x,b,c,d):
    """ Find the maximum of a value and a position on a line 

    Parameters
    ----------
    x: float
        x position on line 1
    c: float
        slope of line 1
    d: float
        y-intercept of line 1
    b: float
        y-intercept of line 2

    Returns
    -------
    float
        the max of a line value and a constant
    """

    one = b
    two = c*x+d
    return np.maximum(one,two)


def dynamics_AScurrent_exp(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ Exponential afterspike current dynamics method takes a current at t0 and returns the current at
    a time step later. 
    """
    return AScurrents_t0*(1.0 + neuron.k*neuron.dt) 
        

def dynamics_AScurrent_vector(neuron, AScurrents_t0, time_step, spike_time_steps, vector):
    """ The vector afterspike current dynamics method keeps track of all of the afterspike
    currents for every previous spike and updates them based on the current time step.
    This method is very slow.

    Parameters
    ----------
    vector : np.ndarray
        an array of all running afterspike current values.
    """
    # an ugly hack to convert lists into numpy arrays
    if isinstance(vector, list):
        vector = neuron.AScurrent_dynamics_method.modify_parameter('vector', np.array)
        
    total = np.zeros(len(vector))
        
    # run through all of the spikes, adding ascurrents based on how long its been since the spike occurred
    for spike_time_step in spike_time_steps:
        try:
            total += vector[:, time_step - spike_time_step]
        except Exception, e:
            pass
        
    return total
    

def dynamics_AScurrent_none(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ This method always returns zeros for the afterspike currents, regardless of input. """
    return np.zeros(len(AScurrents_t0))


def dynamics_voltage_linear(neuron, voltage_t0, AScurrents_t0, inj):
    """ (TODO) Linear voltage dynamics. """

    return voltage_t0 + (inj + np.sum(AScurrents_t0) - neuron.G * neuron.coeffs['G'] * (voltage_t0 - neuron.El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])
    

def dynamics_voltage_quadratic_i_of_v(neuron, voltage_t0, AScurrents_t0, inj, a, b, c, d, e):    
    """ (TODO) Quadratic voltage dynamics equation.

    Parameters
    ----------
    a : float
        constant coefficient of voltage equation
    b : float
        linear coefficient of voltage equation
    c : float
        quadratic coefficient of voltage equation
    d : float
        voltage threshold
    e : float
        voltage used if voltage surpasses threshold (d)
    """
    I_of_v = a + b * voltage_t0 + c * voltage_t0**2  
        
    if voltage_t0 > d:
        I_of_v=e

    return voltage_t0 + (inj + np.sum(AScurrents_t0) - I_of_v) * neuron.dt / (neuron.C * neuron.coeffs['C']) 


def dynamics_voltage_piecewise_linear(neuron, voltage_t0, AScurrents_t0, inj, R_tlparam1, R_tlparam2, R_t1param3, El_tlparam1, El_tlparam2, El_t1param3):
    """ Piecewise linear voltage dynamics methods. This method requires 3 parameters for equations of both resistance and resting potential. 

    Parameters
    ----------
    R_tlparam1 : float 
        TODO
    R_tlparam2 : float
        TODO
    R_tlparam3 : float
        TODO
    El_tlparam1 : float
        TODO
    El_tlparam2 : float
        TODO
    El_tlparam3 : float
        TOOD
    """
    G = 1. / (two_lines(voltage_t0, R_tlparam1, R_tlparam2, R_t1param3))
    El = -two_lines(voltage_t0, El_tlparam1, El_tlparam2, El_t1param3)
    return voltage_t0 + (inj + np.sum(AScurrents_t0) - G * neuron.coeffs['G'] * (voltage_t0 - El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])
    

def dynamics_threshold_adapt_standard(neuron, threshold_t0, voltage_t0, a, b):
    """ Standard adapting threshold dynamics equation.

    Parameters
    ----------
    a : float
        coefficient of voltage
    b : float
        coefficient of threshold
    """

    return threshold_t0 + (a * neuron.coeffs['a'] * (voltage_t0-neuron.El) - 
                           b * neuron.coeffs['b'] * (threshold_t0 - neuron.coeffs['th_inf'] * neuron.th_inf)) * neuron.dt 


def dynamics_threshold_adapt_slow_plus_fast(neuron, threshold_t0, voltage_t0):        
    """The threshold will adapt via two mechanisms: 1. a slower voltage dependent adaptation as in the dynamics_threshold_adapt_standard. 
    These are the components which are fit via optimization and inditial conditions are supplied via the GLM. 2. A fast component initiated 
    by a spike which quickly decays.  These values are estimated via the multi short square stimuli. 
    """
    
    md = neuron.update_method_data

    # initial conditions
    if 'th_spike' not in md:
        md['th_spike'] = [ 0 ]
    if 'th_voltage' not in md:
        md['th_voltage'] = [ neuron.th_inf * neuron.coeffs['th_inf'] ]

    th_spike = md['th_spike'][-1]
    th_voltage = md['th_voltage'][-1] 

    voltage_component = th_voltage + ( md['a_voltage'] * neuron.coeffs['a'] * ( voltage_t0 - neuron.El ) - 
                                       md['b_voltage'] * neuron.coeffs['b'] * ( th_voltage - neuron.coeffs['th_inf'] * neuron.th_inf ) ) * neuron.dt
    spike_component = th_spike - md['b_spike'] * th_spike * neuron.dt

    #------update the voltage and spiking values of the the
    md['th_spike'].append(spike_component)
    md['th_voltage'].append(voltage_component)
    
    return voltage_component+spike_component
    
    
def dynamics_threshold_inf(neuron, threshold_t0, voltage_t0):
    """ Set threshold to the neuron's instantaneous threshold. """
    return neuron.coeffs['th_inf'] * neuron.th_inf


def dynamics_threshold_fixed(neuron, threshold_t0, voltage_t0, value):
    """ Set threshold a fixed constant.

    Parameters
    ----------
    value : float
        fixed constant to use for threshold. 
    """
    return value


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
    """
    new_currents=neuron.asc_vector * neuron.coeffs['asc_vector'] #neuron.asc_vector are amplitudes initiating after the spike is cut
    left_over_currents=AScurrents_t0 * r * np.exp(neuron.k * neuron.dt * neuron.spike_cut_length) #advancing cut currents though the spike    

    return new_currents+left_over_currents


def reset_AScurrent_none(neuron, AScurrents_t0):
    """ Reset afterspike currents to zero. """
    if np.sum(AScurrents_t0)!=0:
        raise Exception('You are running a LIF but the AScurrents are not zero!')
    return np.zeros(len(AScurrents_t0))


def reset_voltage_v_before(neuron, voltage_t0, a, b):
    """ Reset voltage to the previous value with a scale and offset applied.

    Parameters
    ----------
    a : float
        voltage scale constant
    b : float
        voltage offset constant
    """

    return a*(voltage_t0)+b


def reset_voltage_zero(neuron, voltage_t0):
    """ Reset voltage to zero. """
    return 0.0


def reset_voltage_fixed(neuron, voltage_t0, value):
    """ Reset voltage to a fixed value. 

    Parameters
    ----------
    value : float
        the value to which voltage will be reset.
    """
    return value

def reset_threshold_max_v_th(neuron, threshold_t0, voltage_v1, delta):
    """ Return the maximum of threshold and reset voltage offset by a constant. 

    Parameters
    ----------
    delta : float
        value used to offset the return threshold.
    """
    #This is a bit dangerous as it would change if El was not choosen to be zero. Perhaps could change it to absolute value.
    return max(threshold_t0, voltage_v1) + delta  
    
def reset_threshold_inf(neuron, threshold_t0, voltage_v1):
    """ Reset the threshold to instantaneous threshold. """
    return neuron.coeffs['th_inf'] * neuron.th_inf

def reset_threshold_for_adapt_slow_fast(neuron, threshold_t0, voltage_v1):
    '''this method resets voltage and threshold.  Here there are two components a spike (fast)
    component and a slow(component) which getted added 
    '''
    md = neuron.update_method_data
    
    md['th_spike'].append( md['th_spike'][-1] + md['a_spike'] )
    md['th_voltage'].append( md['th_voltage'][-1] ) #note these are the same value.

    return md['th_spike'][-1] + md['th_voltage'][-1]


#: The METHOD_LIBRARY constant groups dynamics and reset methods by group name (e.g. 'voltage_dynamics_method').  Those groups assign each method in this file a string name.  This is used by the GLIFNeuron when initializing its dynamics and reset methods.
METHOD_LIBRARY = {
    'AScurrent_dynamics_method': { 
        'exp': dynamics_AScurrent_exp,
        'exp_ssq': dynamics_AScurrent_exp,
        'exp_glm': dynamics_AScurrent_exp,
        'vector': dynamics_AScurrent_vector,
        'none': dynamics_AScurrent_none
        },
    'voltage_dynamics_method': { 
        'linear': dynamics_voltage_linear,
        'quadratic_i_of_v': dynamics_voltage_quadratic_i_of_v,
        'piecewise': dynamics_voltage_piecewise_linear
        },
    'threshold_dynamics_method': {
        'adapt_sum_slow_fast':  dynamics_threshold_adapt_slow_plus_fast,                        
        'adapt_rebound': dynamics_threshold_adapt_slow_plus_fast,
        'inf': dynamics_threshold_inf,
        'fixed': dynamics_threshold_fixed
        },
    'AScurrent_reset_method': {
        'sum': reset_AScurrent_sum,
        'none': reset_AScurrent_none
        }, 
    'voltage_reset_method': {
        'v_before': reset_voltage_v_before,
        'zero': reset_voltage_zero,
        'fixed': reset_voltage_fixed
        }, 
    'threshold_reset_method': {
        'max_v_th': reset_threshold_max_v_th,
        'inf': reset_threshold_inf,
        'adapt_sum_slow_fast': reset_threshold_for_adapt_slow_fast
        }
}
