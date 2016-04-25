import numpy as np
import logging

from allensdk.internal.model.glif.find_spikes import find_spikes, find_spikes_list

from allensdk.ephys.extract_cell_features import get_stim_characteristics

def least_squares_simple_circuit_fit_RCEl(voltage_list, current_list, dt, no_rest=False):
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
