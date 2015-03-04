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
        :parameter param: the name of the parameter to modify
        :type param: string
        :parameter operator: a function or lambda that returns the desired modified value
        :type operator: callable
        """
        value = operator(self.method.keywords[param])
        self.method.keywords[param] = value
        return value


def two_lines(x,b,c,d):
    """ Find the maximum of a value and a position on a line 

    :parameter x: x position on line 1
    :type x: float
    :parameter c: slope of line 1
    :type c: float
    :parameter d: y-intercept of line 1
    :type d: float
    :parameter b: y-intercept of line 2
    :type b: float
    :returns: the max of a line value and a constant
    :rtype: float
    """
    one = b
    two = c*x+d
    return np.maximum(one,two)


"""
Dynamics Methods
================
These methods are used to update the threshold, voltage, and afterspike currents
of the model at a given time step of the simulation.
"""

"""
Afterspike Current Dynamics Methods
-----------------------------------
These methods expect current afterspike current coefficients, current time step, 
and time steps of all previous spikes to be passed in by the GLIFNeuron. All other function 
parameters must be fixed using the GLIFNeuronMethod class.  They all return an updated
afterspike current array.
"""

def dynamics_AScurrent_exp(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ Exponential afterspike current dynamics method """
    return AScurrents_t0*(1.0 + neuron.k*neuron.dt) 
        

def dynamics_AScurrent_vector(neuron, AScurrents_t0, time_step, spike_time_steps, vector):
    """ The vector afterspike current dynamics method keeps track of all of the afterspike
    currents for every previous spike and updates them based on the current time step.
    This method is very slow.

    :parameter vector: an array of all running afterspike current values.
    :type vector: np.ndarray
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

"""
Voltage Dynamics Methods
------------------------
These methods update the output voltage of the simulation.  They all expect a voltage, 
afterspike current vector, and current injection value to be passed in by the GLIFNeuron. All 
other function parameters must be fixed using the GLIFNeuronMethod class.  They all return an 
updated voltage value.
"""

def dynamics_voltage_linear(neuron, voltage_t0, AScurrents_t0, inj):
    """ Linear voltage dynamics. """
    return voltage_t0 + (inj + np.sum(AScurrents_t0) - neuron.G * neuron.coeffs['G'] * (voltage_t0 - neuron.El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])
    

def dynamics_voltage_quadratic_i_of_v(neuron, voltage_t0, AScurrents_t0, inj, a, b, c, d, e):    
    """ Quadratic voltage dynamics equation.

    :parameter a: constant coefficient of voltage equation
    :type a: float
    :parameter b: linear coefficient of voltage equation
    :type b: float
    :parameter c: quadratic coefficient of voltage equation
    :type c: float
    :parameter d: voltage threshold
    :type d: float
    :parameter e: voltage used if voltage surpasses threshold (d)
    :type e: float
    """
    I_of_v = a + b * voltage_t0 + c * voltage_t0**2  
        
    if voltage_t0 > d:
        I_of_v=e

    return voltage_t0 + (inj + np.sum(AScurrents_t0) - I_of_v) * neuron.dt / (neuron.C * neuron.coeffs['C']) 


def dynamics_voltage_piecewise_linear(neuron, voltage_t0, AScurrents_t0, inj, R_tlparam1, R_tlparam2, R_t1param3, El_tlparam1, El_tlparam2, El_t1param3):
    """ Piecewise linear voltage dynamics methods. This method requires 3 parameters for equations of both resistance and resting potential. 

    :parameter R_tlparam1: 
    :type R_tlparam1: float
    :parameter R_tlparam2: 
    :type R_tlparam2: float
    :parameter R_tlparam3: 
    :type R_tlparam3: float
    :parameter El_tlparam1: 
    :type El_tlparam1: float
    :parameter El_tlparam2: 
    :type El_tlparam2: float
    :parameter El_tlparam3: 
    :type El_tlparam3: float
    """
    G = 1. / (two_lines(voltage_t0, R_tlparam1, R_tlparam2, R_t1param3))
    El = -two_lines(voltage_t0, El_tlparam1, El_tlparam2, El_t1param3)
    return voltage_t0 + (inj + np.sum(AScurrents_t0) - G * neuron.coeffs['G'] * (voltage_t0 - El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])
    

"""
Threshold Dynamics Equations
----------------------------
These methods update the spike threshold of the simulation.  They all expect the current
threshold and voltage values of the simulation to be passed in by the GLIFNeuron. All 
other function parameters must be fixed using the GLIFNeuronMethod class.  They all return an 
updated threshold value.
"""

def dynamics_threshold_adapt_standard(neuron, threshold_t0, voltage_t0, a, b):
    """ Standard adapting threshold dynamics equation.

    :parameter a: coefficient of voltage
    :type a: float
    :parameter b: coefficient of threshold
    :type b: float
    """
    return threshold_t0 + (a * neuron.coeffs['a'] * (voltage_t0-neuron.El) - b * neuron.coeffs['b'] * (threshold_t0 - neuron.coeffs['th_inf'] * neuron.th_inf)) * neuron.dt 
        

def dynamics_threshold_inf(neuron, threshold_t0, voltage_t0):
    """ Set threshold to the neuron's instantaneous threshold. """
    return neuron.coeffs['th_inf'] * neuron.th_inf


def dynamics_threshold_fixed(neuron, threshold_t0, voltage_t0, value):
    """ Set threshold a fixed constant.

    :parameter value: fixed constant to use for threshold. 
    :type value: float
    """
    return value

"""
Reset Methods
=============
These methods are used to reset the threshold, voltage, and afterspike currents
of the model after voltage has surpassed spike threshold.
"""

"""
Afterspike Current Reset Methods
--------------------------------
These methods expect current afterspike current coefficients to be passed in by 
the GLIFNeuron. All other function parameters must be fixed using the GLIFNeuronMethod 
class.  They all return an updated afterspike current array.
"""

def reset_AScurrent_sum(neuron, AScurrents_t0, r):
    """ Reset afterspike currents by adding summed exponentials. 

    :parameter r: a coeffient vector applied to the afterspike currents
    :type r: np.ndarray
    """

    #old way without refrectory period: var_out[2]=neuron.a1*neuron.coeffa1 # a constant multiplied by the amplitude of the excitatory current at reset
    return neuron.asc_vector * neuron.coeffs['asc_vector'] + AScurrents_t0 * r * np.exp(neuron.k * neuron.dt * neuron.spike_cut_length)


def reset_AScurrent_none(neuron, AScurrents_t0):
    """ Reset afterspike currents to zero. """
    if np.sum(AScurrents_t0)!=0:
        raise Exception('You are running a LIF but the AScurrents are not zero!')
    return np.zeros(len(AScurrents_t0))


"""
Voltage Reset Methods
---------------------
These methods update the output voltage of the simulation after voltage has surpassed threshold. 
They all expect a voltageto be passed in by the GLIFNeuron. All other function parameters must be 
fixed using the GLIFNeuronMethod class.  They all return an updated voltage value.
"""

def reset_voltage_v_before(neuron, voltage_t0, a, b):
    """ Reset voltage to the previous value with a scale and offset applied.

    :parameter a: voltage scale constant
    :type a: float
    :parameter b: voltage offset constant
    :type b: float
    """

    return a*(voltage_t0)+b


def reset_voltage_i_v_before(neuron, voltage_t0):
    """ This method is not implemented yet.  It will raise an exception. """
    raise Exception("reset_voltage_IandVbefore not implemented")


def reset_voltage_zero(neuron, voltage_t0):
    """ Reset voltage to zero. """
    return 0.0


def reset_voltage_fixed(neuron, voltage_t0, value):
    """ Reset voltage to a fixed value. 

    :parameter value: the value to which voltage will be reset.
    :type value: float
    """
    return value


"""
Threshold Reset Methods
-----------------------
These methods update the spike threshold of the simulation after a spike has been detected.  
They all expect the current threshold and the reset voltage value of the simulation to be passed in 
by the GLIFNeuron. All other function parameters must be fixed using the GLIFNeuronMethod 
class.  They all return an updated threshold value.
"""    

def reset_threshold_max_v_th(neuron, threshold_t0, voltage_v1, delta):
    """ Return the maximum of threshold and reset voltage offset by a constant. 

    :parameter delta: value used to offset the return threshold.
    :type delta: float
    """
    #This is a bit dangerous as it would change if El was not choosen to be zero. Perhaps could change it to absolute value.
    return max(threshold_t0, voltage_v1) + delta  
    

def reset_threshold_th_before(neuron, threshold_t0, voltage_v1, delta):
    """ Return the previous threshold by a constant. This method is not used and will raise an exception if called.

    :parameter delta: value used to offset the return threshold.
    :type delta: float
    """
    raise Exception ('reset_threshold_th_before should not be called')
    return threshold_t0 + delta


def reset_threshold_inf(neuron, threshold_t0, voltage_v1):
    """ Reset the threshold to instantaneous threshold. """
    return neuron.coeffs['th_inf'] * neuron.th_inf

def reset_threshold_fixed(neuron, threshold_t0, voltage_v1, value):
    """ Reset the threshold to a fixed value. This method is not sued and will raise an exception if called.

    :parameter value: value to return as the reset threshold
    :type value: float
    """
    raise Exception('reset_threshold_fixed should not be called')
    return  value

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
        'adapt_standard': dynamics_threshold_adapt_standard,
        'adapt_rebound': dynamics_threshold_adapt_standard,
        'inf': dynamics_threshold_inf,
        'fixed': dynamics_threshold_fixed
        },
    'AScurrent_reset_method': {
        'sum': reset_AScurrent_sum,
        'none': reset_AScurrent_none
        }, 
    'voltage_reset_method': {
        'v_before': reset_voltage_v_before,
        'i_v_before': reset_voltage_i_v_before,
        'zero': reset_voltage_zero,
        'fixed': reset_voltage_fixed
        }, 
    'threshold_reset_method': {
        'max_v_th': reset_threshold_max_v_th,
        'th_before': reset_threshold_th_before,
        'inf': reset_threshold_inf,
        'adapted': reset_threshold_fixed,
        'fixed': reset_threshold_fixed
        }
}
