import functools
import numpy as np

class GLIFNeuronMethod( object ):
    def __init__(self, method_name, method, method_params):
        self.name = method_name
        self.params = method_params
        self.method = functools.partial(method, **method_params)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    def to_dict(self):
        return {
            'name': self.name,
            'params': self.params
        }

    def modify_parameter(self, param, operator):
        value = operator(self.method.keywords[param])
        self.method.keywords[param] = value
        return value

#------------------------------------------------------------
# dynamics methods
#------------------------------------------------------------

# AScurrent equations 
# all return AScurrents_t1 

def dynamics_AScurrent_exp(neuron, AScurrents_t0, time_step, spike_time_steps):
    return AScurrents_t0*(1.0 + neuron.k*neuron.dt) #calculate each after spike current 
        
def dynamics_AScurrent_vector(neuron, AScurrents_t0, time_step, spike_time_steps, vector):
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
    return np.zeros(len(AScurrents_t0))
   
# voltage dynamics equations 
# all return voltage_t1

def dynamics_voltage_linear(neuron, voltage_t0, AScurrents_t0, inj):
    return voltage_t0 + (inj + np.sum(AScurrents_t0) - neuron.G * neuron.coeffs['G'] * (voltage_t0 - neuron.El)) * neuron.dt / (neuron.C * neuron.coeffs['C'])
    
def dynamics_voltage_quadratic_i_of_v(neuron, voltage_t0, AScurrents_t0, inj, a, b, c, d, e):    
    I_of_v = a + b * voltage_t0 + c * voltage_t0**2  #equation for cell 6 jting
        
    if voltage_t0 > d:
        I_of_v=e

    return voltage_t0+(inj + np.sum(AScurrents_t0)-I_of_v)*neuron.dt/(neuron.C*neuron.coeffs['C']) 
    
# threshold equations 
# all return threshold_t1    

def dynamics_threshold_adapt_standard(neuron, threshold_t0, voltage_t0, a, b):
    return threshold_t0 + (a * neuron.coeffs['a'] * (voltage_t0-neuron.El) - b * neuron.coeffs['b'] * (threshold_t0 - neuron.coeffs['th_inf'] * neuron.th_inf)) * neuron.dt 
        
def dynamics_threshold_inf(neuron, threshold_t0, voltage_t0):
    return neuron.coeffs['th_inf'] * neuron.th_inf

def dynamics_threshold_fixed(neuron, threshold_t0, voltage_t0, value):
    return value

# AScurrent reset rules 
# all return AScurrents_t1

def reset_AScurrent_sum(neuron, AScurrents_t0, t, r):
    #old way without refrectory period: var_out[2]=neuron.a1*neuron.coeffa1 # a constant multiplied by the amplitude of the excitatory current at reset
    # 2:6 are k's
    #return neuron.asc_vector * neuron.coeffs['asc_vector'] + AScurrents_t0 * np.exp(neuron.k * neuron.dt)
    return neuron.asc_vector * neuron.coeffs['asc_vector'] + AScurrents_t0 * r * np.exp(neuron.k * neuron.dt)

def reset_AScurrent_none(neuron, AScurrents_t0, t):
    if np.sum(AScurrents_t0)!=0:
        raise Exception('You are running a LIF but the AScurrents are not zero!')
    return np.zeros(len(AScurrents_t0))

# voltage reset rules
# all return voltage_t1

def reset_voltage_v_before(neuron, voltage_t0, a, b):
    return a*(voltage_t0)+b

def reset_voltage_i_v_before(neuron, voltage_t0):
    raise Exception("reset_voltage_IandVbefore not implemented")

def reset_voltage_zero(neuron, voltage_t0):
    return 0.0

def reset_voltage_fixed(neuron, voltage_t0, value):
    return value
    
# threshold reset rules
# all return threshold_t1

def reset_threshold_max_v_th(neuron, threshold_t0, voltage_v1, delta):
    return max(threshold_t0, voltage_v1) + delta  #This is a bit dangerous as it would change if El was not choosen to be zero. Perhaps could change it to absolute value
    
def reset_threshold_th_before(neuron, threshold_t0, voltage_v1, delta):
    '''it is highly probable that at some point we will need to fit const'''
    '''threshold_t0 and value should be in mV'''
    return threshold_t0 + delta

def reset_threshold_inf(neuron, threshold_t0, voltage_v1):
    '''it is highly probable that at some point we will need to fit const'''
    '''threshold_t0 and value should be in mV'''
    return neuron.coeffs['th_inf'] * neuron.th_inf

def reset_threshold_fixed(neuron, threshold_t0, voltage_v1, value):
    '''it is highly probable that at some point we will need to fit const'''
    '''threshold_t0 and value should be in mV'''
    return  value

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
        'quadratic_i_of_v': dynamics_voltage_quadratic_i_of_v
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
