""" The methods in this module are used for configuring dynamics and reset rules for the GlifNeuron.  
For more details on how to use these methods, see :doc:`glif_models`.
"""
import logging
import functools
import numpy as np
from numpy.distutils.npy_pkg_config import VariableSet

#fast_threshold=[]
#slow_threshold=[]

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


def max_of_line_and_const(x,b,c,d):
    #TODO: move to other library
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

#commenting this out to make sure it doest get called and I forgot to switch is somewhere
#def min_of_line_and_const(x,b,c,d):
#    #TODO: move to other library
#    """ Find the minimum of a value and a position on a line 
#
#    Parameters
#    ----------
#    x: float
#        x position on line 1
#    c: float
#        slope of line 1
#    d: float
#        y-intercept of line 1
#    b: float
#        y-intercept of line 2
#
#    Returns
#    -------
#    float
#        the max of a line value and a constant
#    """
#
#    one = b
#    two = c*x+d
#    return np.minimum(one,two)

def min_of_line_and_zero(x,c,d):
    #TODO: move to other library
    """ Find the minimum of a value and a position on a line 

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

    one = 0
    two = c*x+d
    return np.minimum(one,two)


def dynamics_AScurrent_exp(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ Exponential afterspike current dynamics method takes a current at t0 and returns the current at
    a time step later.
    """

#     print 'HERE: dynamics_threshold_inf'
#    return AScurrents_t0*(1.0 - neuron.k*neuron.dt)
    return AScurrents_t0*np.exp(-neuron.k*neuron.dt)
        

def dynamics_AScurrent_vector(neuron, AScurrents_t0, time_step, spike_time_steps, vector):
    """ The vector afterspike current dynamics method keeps track of all of the afterspike
    currents for every previous spike and updates them based on the current time step.
    This method is very slow.

    Parameters
    ----------
    vector : np.ndarray
        an array of all running afterspike current values.
    """
    # a hack to convert lists into numpy arrays
    if isinstance(vector, list):
        vector = neuron.AScurrent_dynamics_method.modify_parameter('vector', np.array)
        
    total = np.zeros(len(vector))
        
    # run through all of the spikes, adding ascurrents based on how long its been since the spike occurred
    for spike_time_step in spike_time_steps:
        try:
            total += vector[:, time_step - spike_time_step]
        except Exception as e:
            pass
        
    return total
    

def dynamics_AScurrent_none(neuron, AScurrents_t0, time_step, spike_time_steps):
    """ This method always returns zeros for the afterspike currents, regardless of input. """
    return np.zeros(len(AScurrents_t0))


def dynamics_voltage_linear_forward_euler(neuron, voltage_t0, AScurrents_t0, inj):
    """ (TODO) Linear voltage dynamics. """
    return voltage_t0 + (inj + np.sum(AScurrents_t0) - neuron.G * neuron.coeffs['G'] * (voltage_t0 - neuron.El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])

def dynamics_voltage_linear_exact(neuron, voltage_t0, AScurrents_t0, inj):
    """ (TODO) Linear voltage dynamics. """

    C = (neuron.C * neuron.coeffs['C'])
    I = inj + np.sum(AScurrents_t0)
    g = neuron.G * neuron.coeffs['G']
    tau = g/C
    N = (I+ g*neuron.El)/C

    return voltage_t0*np.exp(-neuron.dt*tau) + N*(1-np.exp(-tau*neuron.dt))/tau

def dynamics_voltage_piecewise_linear_exact(neuron, voltage_t0, AScurrents_t0, inj, R_tlparam1, R_tlparam2, R_t1param3, El_slope_param, El_intercept_param):
    """ (TODO) Linear voltage dynamics. """

    C = (neuron.C * neuron.coeffs['C'])
    I = inj + np.sum(AScurrents_t0)
    g = neuron.coeffs['G']/max_of_line_and_const(voltage_t0, R_tlparam1, R_tlparam2, R_t1param3)
    tau = g/C
    El = min_of_line_and_zero(voltage_t0, El_slope_param, El_intercept_param)
    N = (I+ g*El)/C
    
    G = 1. / (max_of_line_and_const(voltage_t0, R_tlparam1, R_tlparam2, R_t1param3))
    El = min_of_line_and_zero(voltage_t0, El_slope_param, El_intercept_param)

    return voltage_t0*np.exp(-neuron.dt*tau) + N*(1-np.exp(-tau*neuron.dt))/tau
    

def dynamics_voltage_quadratic_forward_euler(neuron, voltage_t0, AScurrents_t0, inj, a, b, c, d, e):    
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


def dynamics_voltage_piecewise_linear_forward_euler(neuron, voltage_t0, AScurrents_t0, inj, R_tlparam1, R_tlparam2, R_t1param3, El_slope_param, El_intercept_param):
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
    #confirm this is being implimented correctly.
    G = 1. / (max_of_line_and_const(voltage_t0, R_tlparam1, R_tlparam2, R_t1param3))
#    El = min_of_line_and_const(voltage_t0, El_tlparam1, El_tlparam2, El_t1param3) #this is for when El wasnt forced to zero
    El = min_of_line_and_zero(voltage_t0, El_slope_param, El_intercept_param)

    return voltage_t0 + (inj + np.sum(AScurrents_t0) - G * neuron.coeffs['G'] * (voltage_t0 - El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])
    

def dynamics_threshold_adapt_standard(neuron, threshold_t0, voltage_t0, AScurrents_t0, inj, a, b):
    """ Standard adapting threshold dynamics equation.

    Parameters
    ----------
    neuron : class
    a : float
        coefficient of voltage
    b : float
        coefficient of threshold
    AScurrents_t0 : not used here 
    inj : not used here
    """
#     print 'HERE: dynamics_threshold_adapt_standard'
    return threshold_t0 + (a * neuron.coeffs['a'] * (voltage_t0-neuron.El) - 
                           b * neuron.coeffs['b'] * (threshold_t0 - neuron.coeffs['th_inf'] * neuron.th_inf)) * neuron.dt 

#TODO: depricate when sure that dynamics_threshold_three_sep_components will be the official function
#def dynamics_threshold_adapt_slow_plus_fast(neuron, threshold_t0, voltage_t0):        
#    """The threshold will adapt via two mechanisms: 1. a slower voltage dependent adaptation as in the dynamics_threshold_adapt_standard. 
#    These are the components which are fit via optimization and inditial conditions are supplied via the GLM. 2. A fast component initiated 
#    by a spike which quickly decays.  These values are estimated via the multi short square stimuli. 
#    """
#    
#    md = neuron.update_method_data
#
#    # initial conditions
#    if 'th_spike' not in md:
#        md['th_spike'] = [ 0 ]
#    if 'th_voltage' not in md:
#        md['th_voltage'] = [ neuron.th_inf * neuron.coeffs['th_inf'] ]
#
#    th_spike = md['th_spike'][-1]
#    th_voltage = md['th_voltage'][-1] 
#
##     print 'a_spike', md['a_spike']
##     print 'b_spike', md['b_spike']
##     print 'a_voltage', md['a_voltage']
##     print 'b_voltage', md['b_voltage']
#
#    voltage_component = th_voltage + ( md['a_voltage'] * neuron.coeffs['a'] * ( voltage_t0 - neuron.El ) - 
#                                       md['b_voltage'] * neuron.coeffs['b'] * ( th_voltage - neuron.coeffs['th_inf'] * neuron.th_inf ) ) * neuron.dt
#    spike_component = th_spike - md['b_spike'] * th_spike * neuron.dt
#
#    
#    #------hack to plot slow and fast component
##    global fast_threshold
##    global slow_threshold
##    fast_threshold.append(spike_component)
##    slow_threshold.append(voltage_component)
#    
#    #------update the voltage and spiking values of the the
#    md['th_spike'].append(spike_component)
#    md['th_voltage'].append(voltage_component)
#    
#    return voltage_component+spike_component

#TODO: deprecate when confirm the forward and exact equations are doing the right thing.
# def dynamics_threshold_three_sep_components_hybrid(neuron, threshold_t0, voltage_t0, AScurrents_t0, inj):        
#     """This is the sequel to the dynamics_threshold_adapt_slow_plus_fast module. Here as in 
#     dynamics_threshold_adapt_slow_plus_fast. The threshold will adapt via two mechanisms: 1. a 
#     slower voltage dependent adaptation as in the dynamics_threshold_adapt_standard. and 2. a 
#     fast component initiated by a spike which quickly decays.  These two component are 
#     in reference to zero and are recorded in the md parameters.  The third component refers to 
#     th_inf which is added on in the end as opposed to being encompassed in the voltage component 
#     of the threshold which caused a problem in optimizing th_inf.
#     AScurrents_t0 : not used here 
#     inj : not used here
#     """
#     md = neuron.update_method_data
# 
#     # initial conditions
#     if 'th_spike' not in md:
#         md['th_spike'] = [ 0 ]
#         logging.warning('delete this function it is a hybrid')
#     if 'th_voltage' not in md:
#         md['th_voltage'] = [ 0 ]
# 
#     th_spike = md['th_spike'][-1]
#     th_voltage = md['th_voltage'][-1] 
# 
#     #why are this variables being saved in an md variable?
#     #TODO: HOW CAN NEURON.COEFFS['A'] BE MULTIPLIED AFTERWARD SINCE IT IS ALSO MULTIPLYING THE 
#     #VOLTAGE COMPONENT.
#     voltage_component = th_voltage + ( md['a_voltage'] *  ( voltage_t0 - neuron.El ) - 
#                                        md['b_voltage'] * neuron.coeffs['b'] * ( th_voltage ) ) * neuron.dt
#     
#     #spike_component = th_spike - md['b_spike'] * th_spike * neuron.dt
#     spike_component = th_spike*np.exp(-md['b_spike'] * neuron.dt)
# 
#     #------update the voltage and spiking values of the the
#     md['th_spike'].append(spike_component)
#     md['th_voltage'].append(voltage_component)
#     
#     #TODO: I dont trust the a_coeff being multiplied by the whole function
#     return neuron.coeffs['a']*voltage_component+spike_component+neuron.th_inf * neuron.coeffs['th_inf']    

def spike_component_of_threshold_forward_euler(th_t0, b_spike, dt):
    '''Spike component of threshold modeled as an exponential decay. Implemented 
    here for forward Euler 
    
    Parameters
    ----------
    th_t0 : float
        threshold input to function
    b_spike : float
        decay constant of exponential
    dt : float
        time step
    '''
    b_spike=-b_spike #TODO: this is here because b_spike is always input as positive although it is negative
    return th_t0 + th_t0*b_spike * dt                                                                         
    
def spike_component_of_threshold_exact(th0, b_spike, t):
    '''Spike component of threshold modeled as an exponential decay. Implemented 
    here as exact analytical solution. 
    
    Parameters
    ----------
    th0 : float
        threshold input to function
    b_spike : float
        decay constant of exponential
    t : float or array
        time step if used in an Euler setup
        time if used analytically
    '''
    b_spike=-b_spike
    return th0*np.exp(b_spike * t)

def voltage_component_of_threshold_forward_euler(th_t0, v_t0, dt, a_voltage, b_voltage, El):
    '''Equation 2.1 of Mihalas and Nieber, 2009 implemented for use in forward Euler. Note 
    here all variables are in reference to threshold infinity.  Therefore thr_inf is zero 
    here (replaced threshold_inf with 0 in the equation to be verbose). This is done so that 
    th_inf can be optimized without affecting this function. 
    
    Parameters
    ----------
    th_t0 : float
        threshold input to function
    v_t0 : float
        voltage input to function
    dt : float
        time step
    a_voltage : float
        constant a
    b_voltage : float
        constant b
    El : float
        reversal potential
    '''
    return th_t0 + (a_voltage*(v_t0-El)-b_voltage*(th_t0-0))*dt

def voltage_component_of_threshold_exact(th0, v0, I, t, a_voltage, b_voltage, C, g, El):
    '''Note this function is the exact formulation; however, dt is used because t0 is the initial time and dt
    is the time the function is exactly evaluated at. Note: that here, this equation is in reference to th_inf.
    Therefore th0 is the total threshold-thr_inf (threshold_inf replaced with 0 in the equation to be verbose).  
    This is done so that th_inf can be optimized without affecting this function.
    
    Parameters
    ----------
    th0 : float
        threshold input to function
    v0 : float
        voltage input to function
    I : float
        total current entering neuron (note if there are after spike currents these must be included in this value)
    t : float or array
        time step if used in an Euler setup
        time if used analytically
    a_voltage : float
        constant a
    b_voltage : float
        constant b
    C : float
        capacitance
    g : float
        conductance (1/resistance)
    El : float 
        reversal potential
    '''
    beta=(I+g*El)/g
    phi=a_voltage/(b_voltage-g/C)
    return phi*(v0-beta)*np.exp(-g*t/C)+1/(np.exp(b_voltage*t))*(th0-phi*(v0-beta)-
                        (a_voltage/b_voltage)*(beta-El)-0) +(a_voltage/b_voltage)*(beta-El) +0


def dynamics_threshold_three_components_forward(neuron, threshold_t0, voltage_t0, AScurrents_t0, inj, 
                                                a_spike, b_spike, a_voltage, b_voltage):
    """Threshold dynamics implemented for forward Euler. The threshold will adapt via two mechanisms: 
    1. a voltage dependent adaptation.  
    2. a component initiated by a spike which decays as an exponential.  
    These two component are in reference to threshold infinity and are recorded 
    in the neuron's threshold components.
    The third component refers to th_inf which is added separately as opposed to being 
    included in the voltage component of the threshold as is done in equation 2.1 of
    Mihalas and Nieber 2009.  Threshold infinity is removed for simple optimization.
    
    Parameters
    ----------
    neuron : class
    threshold_t0 : float
        threshold input to function
    voltage_t0 : float
        voltage input to function
    AScurrents_t0 : not used here 
    inj : not used here
    """
    tcs = neuron.get_threshold_components()
    
    th_spike = tcs['spike'][-1]
    th_voltage = tcs['voltage'][-1] 

    a_voltage = a_voltage * neuron.coeffs['a']
    b_voltage = b_voltage * neuron.coeffs['b']

    spike_component = spike_component_of_threshold_forward_euler(th_t0, b_spike, neuron.dt)    
    voltage_component = voltage_component_of_threshold_forward_euler(th_voltage, voltage_t0, dt, a_voltage, b_voltage, neuron.El)

    neuron.add_threshold_components(spike_component, voltage_component)
    
    return voltage_component+spike_component+neuron.th_inf * neuron.coeffs['th_inf']    


def dynamics_threshold_three_components_exact(neuron, threshold_t0, voltage_t0, AScurrents_t0, inj,
                                              a_spike, b_spike, a_voltage, b_voltage):
    """Analitical solution for threshold dynamics. The threshold will adapt via two mechanisms: 
    1. a voltage dependent adaptation.  
    2. a component initiated by a spike which decays as an exponential.  
    These two component are in reference to threshold infinity and are recorded 
    in the neuron's threshold components.
    The third component refers to th_inf which is added separately as opposed to being 
    included in the voltage component of the threshold as is done in equation 2.1 of
    Mihalas and Nieber 2009.  Threshold infinity is removed for simple optimization.
    
    Parameters
    ----------
    neuron : class
    threshold_t0 : float
        threshold input to function
    voltage_t0 : float
        voltage input to function
    AScurrents_t0 : vector
        values of after spike currents 
    inj : float
        current injected into the neuron
    """
    tcs = neuron.get_threshold_components()

    th_spike = tcs['spike'][-1]
    th_voltage = tcs['voltage'][-1] 

    a_voltage = a_voltage * neuron.coeffs['a']
    b_voltage = b_voltage * neuron.coeffs['b']

    I = inj + np.sum(AScurrents_t0)
    C = neuron.C * neuron.coeffs['C']
    g = neuron.G * neuron.coeffs['G']

    voltage_component=voltage_component_of_threshold_exact(th_voltage, voltage_t0, I, neuron.dt, a_voltage, b_voltage, C, g, neuron.El)
    spike_component = spike_component_of_threshold_exact(th_spike, b_spike, neuron.dt)
 
    #------update the voltage and spiking values of the the
    neuron.add_threshold_components(spike_component, voltage_component)
     
    return voltage_component+spike_component+neuron.th_inf * neuron.coeffs['th_inf']    

def dynamics_threshold_spike_component(neuron, threshold_t0, voltage_t0, AScurrents_t0, inj,
                                       a_spike, b_spike, a_voltage, b_voltage):
    """Analytical solution for spike component of threshold. The threshold will adapt via 
    a component initiated by a spike which decays as an exponential.  The component is in 
    reference to threshold infinity and are recorded in the neuron's threshold components.  The voltage 
    component of the threshold is set to zero in the threshold components  because it is zero here  
    The third component refers to th_inf which is added separately as opposed to being 
    included in the voltage component of the threshold as is done in equation 2.1 of
    Mihalas and Nieber 2009.  Threshold infinity is removed for simple optimization.
    
    Parameters
    ----------
    neuron : class
    threshold_t0 : float
        threshold input to function
    voltage_t0 : float
        voltage input to function
    AScurrents_t0 : vector
        values of after spike currents 
    inj : float
        current injected into the neuron
    """

    tcs = neuron.get_threshold_components()

    th_spike = tcs['spike'][-1]
    th_voltage = tcs['voltage'][-1] 

    spike_component = spike_component_of_threshold_exact(th_spike, b_spike, neuron.dt)
 
    #------update the voltage and spiking values of the the
    neuron.add_threshold_components(spike_component, 0.0)
     
    return spike_component+neuron.th_inf * neuron.coeffs['th_inf']


def dynamics_threshold_inf(neuron, threshold_t0, voltage_t0, AScurrents_t0, inj):
    """ Set threshold to the neuron's instantaneous threshold. 

    Parameters
    ----------
    neuron : class
    threshold_t0 : not used here
    voltage_t0 : not used here
    AScurrents_t0 : not used here 
    inj : not used here
    AScurrents_t0 : not used here 
    inj : not used here
    """   
    return neuron.coeffs['th_inf'] * neuron.th_inf


def reset_AScurrent_sum(neuron, AScurrents_t0, r):
    """ Reset afterspike currents by adding summed exponentials. Left over currents from last spikes as 
    well as newly initiated currents from current spike.  Currents amplitudes in neuron.asc_amp_array need
    to be the amplitudes advanced though the spike cutting.  I.e. In the preprocessor if the after spike currents 
    are calculated via the GLM from spike initiation the amplitude at the time after the spike cutting needs to 
    be calculated and neuron.asc_amp_array needs to be set to this value.
    
    Parameters
    ----------
    r : np.ndarray
        a coefficient vector applied to the afterspike currents
    """
    new_currents=neuron.asc_amp_array * neuron.coeffs['asc_amp_array'] #neuron.asc_amp_array are amplitudes initiating after the spike is cut
    left_over_currents=AScurrents_t0 * r * np.exp(-(neuron.k * neuron.dt * neuron.spike_cut_length)) #advancing cut currents though the spike    

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


def reset_voltage_i_v_before(neuron, voltage_t0):
    """ This method is not implemented yet.  It will raise an exception. """
    raise Exception("reset_voltage_IandVbefore not implemented")


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
    

def reset_threshold_th_before(neuron, threshold_t0, voltage_v1, delta):
    """ Return the previous threshold by a constant. This method is not used and will raise an exception if called.

    Parameters
    ----------
    delta : float
        value used to offset the return threshold.
    """
    raise Exception ('reset_threshold_th_before should not be called')
    return threshold_t0 + delta


def reset_threshold_inf(neuron, threshold_t0, voltage_v1):
    """ Reset the threshold to instantaneous threshold. """
    return neuron.coeffs['th_inf'] * neuron.th_inf


def reset_threshold_fixed(neuron, threshold_t0, voltage_v1, value):
    """ Reset the threshold to a fixed value. This method is not sued and will raise an exception if called.

    Parameters
    ----------
    value : float
        value to return as the reset threshold
    """
    raise Exception('reset_threshold_fixed should not be called')
    return  value


def reset_threshold_three_components(neuron, threshold_t0, voltage_v1, a_spike):
    '''This method resets the two components of the threshold: a spike (fast)
    component and a (voltage) component which are summed. 
    '''
    tcs = neuron.get_threshold_components()
    neuron.add_threshold_components( tcs['spike'][-1] + a_spike,  #adding spiking component of threshold ontop of already existent spike component.
                                     tcs['voltage'][-1] ) #note these are the same value.
    
    return tcs['spike'][-1] + tcs['voltage'][-1] + neuron.th_inf * neuron.coeffs['th_inf']


#: The METHOD_LIBRARY constant groups dynamics and reset methods by group name (e.g. 'voltage_dynamics_method').  
#Those groups assign each method in this file a string name.  This is used by the GlifNeuron when initializing 
#its dynamics and reset methods.
METHOD_LIBRARY = {
    'AScurrent_dynamics_method': { 
        'exp':    dynamics_AScurrent_exp,
        'vector': dynamics_AScurrent_vector,
        'none':   dynamics_AScurrent_none
        },
    'voltage_dynamics_method': { 
        'linear_forward_euler':           dynamics_voltage_linear_forward_euler,
        'quadratic_forward_euler':        dynamics_voltage_quadratic_forward_euler,
        'piecewise_linear_forward_euler': dynamics_voltage_piecewise_linear_forward_euler,
        'linear_exact':                   dynamics_voltage_linear_exact,
        'piecewise_linear_exact':         dynamics_voltage_piecewise_linear_exact,
        },
    'threshold_dynamics_method': {
        'spike_component':          dynamics_threshold_spike_component, 
        'inf':                      dynamics_threshold_inf,
        'three_components_forward': dynamics_threshold_three_components_forward, 
        'three_components_exact':   dynamics_threshold_three_components_exact 
        },
    'AScurrent_reset_method': {
        'sum':  reset_AScurrent_sum,
        'none': reset_AScurrent_none
        }, 
    'voltage_reset_method': {
        'v_before':   reset_voltage_v_before,
        'i_v_before': reset_voltage_i_v_before,
        'zero':       reset_voltage_zero,
        'fixed':      reset_voltage_fixed
        }, 
    'threshold_reset_method': {
        'max_v_th':         reset_threshold_max_v_th,
        'th_before':        reset_threshold_th_before,
        'inf':              reset_threshold_inf,
        'fixed':            reset_threshold_fixed,
        'three_components': reset_threshold_three_components
        }
}
