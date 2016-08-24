import numpy as np
import logging

from allensdk.model.glif.find_spikes import find_spikes, find_spikes_list

from allensdk.ephys.extract_cell_features import get_stim_characteristics

def smooth(voltage, current, filter_size_s, dt):
    '''Smooths voltage, current, and dV/dt for use in least squares R, C, El calculation.
    Note that currently there is still a bit of question about the order of the convolution
    operation and how similar it should be if the smoothed dv/dt is calculated by first 
    smoothing the voltage and then calculating dV/dt or by calculating dV/dt first and then 
    smoothing.  In practice these numbers are very similar (within e-12) however that 
    difference is still intriguing.  Please see experimental smooth function in 
    verify_rcel_GLM.py (in the GLIF_clean verification folder)
    
    inputs:
        voltage: numpy array of voltage time series
        current: numpy array of current time series
        filter_size_s: float of the size of the filter in seconds
        dt: float size of time step in time series
    outputs:
        sm_v: numpy array of voltage time series smoothed with the filter
        sm_i: numpy array of current time series smoothed with the filter
        sm_dvdt: numpy array of v time series smoothed with the filter
    '''
    #Square filter for smoothing
    sq_filt = np.ones((round(filter_size_s/dt,1)))
    sq_filt = sq_filt/len(sq_filt)
    
    #take diff and then smooth
    i = current[0:len(voltage)-1]
    v = voltage[0:len(voltage)-1]
    vs = voltage[1:len(voltage)]   
    
    #smooth data and diff
    sm_v = np.convolve(v,sq_filt,'same')
    sm_vs = np.convolve(vs,sq_filt,'same')
    sm_i = np.convolve(i,sq_filt,'same')
    sm_dvdt = (sm_vs-sm_v)/dt

    return sm_v, sm_i, sm_dvdt


def least_squares_simple_circuit_with_smoothing_fit_RCEl(voltage_list, current_list, dt, filter_size, no_rest=False):
    '''Calculate resistance, capacitance and resting potential by performing 
    least squares on a smoothed current and voltage.
    inputs:
        voltage_list: list of voltage responses for several sweep repeats
        current_list: list of current injections for several sweep repeats
        dt: time step size
    outputs:
        list of capacitance, resistance and resting potential values for each sweep
    '''
    r_list=[]
    c_list=[]
    El_list=[]
    for voltage, current in zip(voltage_list, current_list):
        sm_v, sm_i, sm_dvdt=smooth(voltage, current, filter_size, dt)
        matrix_sm=np.ones((len(sm_v), 3))
        matrix_sm[:,0]=sm_v
        matrix_sm[:,1]=sm_i
        
        lsq_der_sm=np.linalg.lstsq(matrix_sm, sm_dvdt)[0] 
        C=1/lsq_der_sm[1]
        R=-1/(C*lsq_der_sm[0])
        El=C*R*lsq_der_sm[2]
        
        r_list.append(R)
        c_list.append(C)
        El_list.append(El)
    return R, C, El

def least_squares_simple_circuit_with_NOT_smoothing_fit_RCEl(voltage_list, current_list, dt, no_rest=False):
    '''Calculate resistance, capacitance and resting potential by performing 
    least squares on a smoothed current and voltage.
    inputs:
        voltage_list: list of voltage responses for several sweep repeats
        current_list: list of current injections for several sweep repeats
        dt: time step size
    outputs:
        list of capacitance, resistance and resting potential values for each sweep
    '''
    r_list=[]
    c_list=[]
    El_list=[]
    for voltage, current in zip(voltage_list, current_list):
        i = current[0:len(voltage)-1]
        v = voltage[0:len(voltage)-1]
        vs = voltage[1:len(voltage)]   
        dvdt = (vs-v)/dt
        
        matrix=np.ones((len(v), 3))
        matrix[:,0]=v
        matrix[:,1]=i
        
        lsq_der_sm=np.linalg.lstsq(matrix, dvdt)[0] 
        C=1/lsq_der_sm[1]
        R=-1/(C*lsq_der_sm[0])
        El=C*R*lsq_der_sm[2]
        
        r_list.append(R)
        c_list.append(C)
        El_list.append(El)
    return R, C, El

def least_squares_simple_circuit_fit_RCEl(voltage_list, current_list, dt, no_rest=False):
#NOTE THIS IS THE VERSION THAT IS CURRENTLY IN THE PIPELINE
    '''Calculate resistance, capacitance and resting potential by performing 
    least squares on current and voltage.
    inputs:
        voltage_list: list of voltage responses for several sweep repeats
        current_list: list of current injections for several sweep repeats
        dt: time step size
    outputs:
        list of capacitance, resistance and resting potential values for each sweep
    '''
    capacitance_list=[]
    resistance_list=[]
    El_list=[]
    for voltage, current in zip(voltage_list, current_list): 
        spikes, _ = find_spikes_list([voltage], dt)
        if len(spikes[0]) > 0:
            logging.warning('There is a spike in your subthreshold noise. However continuing using least squares')
        if no_rest:
            #find region of stimulus. Note this will only work if there is no test pulse and only one stimulus injection (i.e. entire noise sweep is already truncated to just low amplitude noise injection)
            t = np.arange(0, len(current)) * dt
            (_, _, _, start_idx, end_idx) = get_stim_characteristics(current, t, no_test_pulse=True)
            stim_dur = end_idx - start_idx
            stim_start = start_idx

            voltage = voltage[stim_start:stim_start+stim_dur]
            current = current[stim_start:stim_start+stim_dur]   

        v_nplus1=voltage[1:]
        voltage=voltage[0:-1]
        current=current[0:-1]
        matrix=np.ones((len(voltage), 3))
        matrix[:,0]=voltage
        matrix[:,1]=current
        out=np.linalg.lstsq(matrix, v_nplus1)[0] 
        
        capacitance=dt/out[1]
        resistance=dt/(capacitance*(1-out[0]))
        El=(capacitance*resistance*out[2])/dt
           
        capacitance_list.append(capacitance)
        resistance_list.append(resistance)
        El_list.append(El)    
        
#    print "R via least squares", np.mean(resistance_list)*1e-6, "Mohms"
#    print "C via least squares", np.mean(capacitance_list)*1e12, "pF"
#    print "El via least squares", np.mean(El_list)*1e3, "mV" 
#    print "tau", np.mean(resistance_list)*np.mean(capacitance_list)*1e3, "ms"
    
    return resistance_list, capacitance_list, El_list


def least_squares_simple_circuit_fit_REl(voltage_list, current_list, Cap, dt):
    '''Calculate resistance and resting potential by performing 
    least squares on current and voltage.
    inputs:
        voltage_list: list of voltage responses for several sweep repeats
        current_list: list of current injections for several sweep repeats
        Cap: capacitance (float) 
        dt: time step size
    outputs:
        list of resistance and resting potential values for each sweep
    '''
    resistance_list=[]
    El_list=[]
    for voltage, current in zip(voltage_list, current_list):    
        v_nplus1=voltage[1:]
        voltage=voltage[0:-1]
        current=current[0:-1]
        matrix=np.ones((len(voltage), 2))
        matrix[:,0]=voltage
        out=np.linalg.lstsq(matrix, v_nplus1-(current*dt)/Cap)[0] 
        
        resistance=dt/(Cap*(1-out[0]))
        El=(Cap*resistance*out[1])/dt        
        
        resistance_list.append(resistance)
        El_list.append(El)    
    
#    print "R via least squares", np.mean(resistance_list)*1e-6, "Mohms"
#    print "C via least squares", np.mean(capacitance_list)*1e12, "pF"
#    print "El via least squares", np.mean(El_list)*1e3, "mV" 
#    print "tau", np.mean(resistance_list)*np.mean(capacitance_list)*1e3, "ms"
    
    return resistance_list, El_list

def fit_RmReCEL_lssq(subthresh_noise_voltage_list, subthresh_noise_current_list, dt, Rs):
    '''Uses least squares on subthreshold noise to fit the membrane resistance (Rm), the electrode (Re), 
    the membrane capacitance C, and the resting potential El of a neuron model with an electrode.  The neuron is in 
    parallel with a seal resistance coming from the electrode. This whole circuit is in series with an 
    electrode resistance.  Note that this assumes the electrode capacitance is compensated for.
    
    Inputs
        subthresh_noise_voltage_list: list of numpy arrays
            recorded (observed) voltage, each sweep in an array
        current: list of numpy array
            stimulus injected though electrode, each sweep in an array
        dt: float
            time step
        Rs: float
            electrode seal resistance
    Outputs
        Rm_list: list of floats 
            Membrane resistance
        Re_list: list of floats
            Electrode resisitance
        C_list: list of floats
            Membrane capacitance
        El_list: list of floats
            Resting potential
            '''
    
    Rm_list=[] 
    Re_list=[] 
    C_list=[]
    El_list=[]
    for voltage, current in zip(subthresh_noise_voltage_list, subthresh_noise_current_list):
        i=current[0:-1]
        iplus1=current[1:]
        Vo=voltage  # observed voltage
        #--set up matrix for linear regression.  Note that matrix for derivative versus non derivative regression is the same.
        matrix=np.ones((len(Vo)-1, 4))
        matrix[:,0]=Vo[0:len(Vo)-1]
        matrix[:,2]=i
        matrix[:,3]=iplus1-i
        #--do least squares
        out_of_lssq=np.linalg.lstsq(matrix, Vo[1:]-Vo[:-1])[0] 
        
        #--solve for matrix coefficients
        Re=out_of_lssq[3]
        A=np.zeros((3,3))
        A[0,0]=Re
        A[1,0]=1.
        A[0,1]=1. 
        A[2,2]=1.
        
        b=np.zeros((3,1))
        b[0,0]=out_of_lssq[2]
        b[1,0]=-out_of_lssq[0]
        b[2,0]=out_of_lssq[1]
            
        x=np.linalg.solve(A, b) 
        C=dt/x[1]
        Rm=1./(x[0]*C/dt -1/Rs)
        El=x[2]*C*Rm/dt

        Rm_list.append(Rm.tolist()[0]) 
        Re_list.append(Re)
        C_list.append(C.tolist()[0])
        El_list.append(El.tolist()[0])
        
    return Rm_list, Re_list, C_list, El_list
    
    
