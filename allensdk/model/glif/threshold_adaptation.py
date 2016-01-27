import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import logging
import matplotlib.pyplot as plt
import logging
THRESH_PCT_MULTIBLIP = 0.05

from allensdk.model.glif.find_spikes import find_spikes_list

def calc_spike_component_of_threshold_from_multiblip(multi_SS, dt, MAKE_PLOT=False, SHOW_PLOT=False, BLOCK=False, PUBLICATION_PLOT=False):
    '''Calculate the spike components of the threshold by fitting a decaying exponential function to data to threshold versus time 
    since last spike in the multiblip data. The exponential is forced to decay to the local th_inf (calculated as the mean all of the 
    threshold values of the first spikes in each individual triblip stimulus). 
    
    Notes: The standard SDK spike detection algorithm does not work with the multiblip stimulus due to artifacts 
    when the stimulus turns on and off. Please see the find_multiblip_spikes module for more information.
    
    Input
    
    multi_SS: dictionary
        contains multiblip information such as current and stimulus
    dt: float
        time step in seconds
    
    Returns
    
    const_to_add_to_thresh_for_reset: float
        amplitude of the exponential fit otherwise known as a_spike
    decay_const: float
        decay constant of exponential. Note the function fit is a negative exponential which will mean this value will 
        either have to be negated when it is used or the functions used will have to have to include the negative.
    thresh_inf: float
    
    '''    
    multi_SS_v=multi_SS['voltage']
    multi_SS_i=multi_SS['current']

    # get indicies of spikes
    spike_ind = find_multiblip_spikes(multi_SS_i, multi_SS_v, dt)
#    spike_ind, _=find_spikes_list(multi_SS_v, dt)

    # eliminate spurious spikes that may exist
    spike_lt=[np.where(SI<int(2.02/dt))[0] for SI in spike_ind]
    if len(np.concatenate(spike_lt))>0:
        logging.warning('there is a spike before the stimulus in the multiblip')
    spike_ind=[np.delete(SI,ind)for SI, ind in zip(spike_ind, spike_lt)]
    spike_gt=[np.where(SI>int(2.1/dt))[0] for SI in spike_ind]
    if len(np.concatenate(spike_gt))>0:
        logging.warning('there is a spike after the stimulus in the multiblip')    
    spike_ind=[np.delete(SI,ind)for SI, ind in zip(spike_ind, spike_gt)]
    
    # intialize output lists
    time_previous_spike=[]
    threshold=[]
    thresh_first_spike=[]  #will set constant to this
            # voltage at all spikes in a single tri blip

    if MAKE_PLOT:
        plt.figure()
        
    # loop though each tri blip stimulus in muliblip stimulus
    for k in range(0, len(multi_SS_v)):
        thresh=[multi_SS_v[k][j] for j in spike_ind[k]] # voltage at all spikes in a single tri blip
        if thresh!=[] and len(thresh)>1:# there needs to be more than one spike so that we can find the time difference
            thresh_first_spike.append(thresh[0])
            threshold.append(thresh[1:])
            time_before_temp=[]
            for j in range(1,len(thresh)):
                time_before_temp.append((spike_ind[k][j]-spike_ind[k][j-1])*dt)
            #for each spike calculate the time from the previous spike   
            time_previous_spike.append(time_before_temp)    
        if MAKE_PLOT:
            plt.subplot(2,1,1)
            plt.plot(np.arange(0, len(multi_SS_i[k]))*dt, multi_SS_i[k]*1e12, lw=2)
            plt.ylabel('current (pA)', fontsize=16)
            plt.xlim([2., 2.12])
            plt.title('Triple Short Square', fontsize=20)
            plt.subplot(2,1,2)
            plt.plot(np.arange(0, len(multi_SS_v[k]))*dt, multi_SS_v[k], lw=2)
            plt.plot(spike_ind[k]*dt, thresh, '.k', ms=16)
            plt.ylabel('voltage (V)', fontsize=16)
            plt.xlabel('time (s)', fontsize=16)
            plt.xlim([2., 2.12])
    
    if SHOW_PLOT:        
        plt.show(block=False)

    # put numbers into one vector for fitting of exponential function
    thresh_inf=np.mean(thresh_first_spike)  #note this threshold infinity isnt the one coming from single blip
    try: #this try here because sometimes even though have the traces there isnt more than one trace with two spikes
        threshold=np.concatenate(threshold)
        time_previous_spike=np.concatenate(time_previous_spike)  #note that this will have nans in it
    
        if MAKE_PLOT:
            plt.figure()
            plt.plot(time_previous_spike, threshold, '.k', ms=16)
            plt.ylabel('threshold (mV)')
            plt.xlabel('time since last spike (s)')
    
        # calculate values of exponential function both if force function to local threshold infinity and not forcing to a value
        # (not forcing to a value seems less valid unless a bunch of points are added corresponding to the threshold of the
        # first spike at time equal infinity (because the first spike is a spike that happens where the spike before it was an
        # infinite time away)).  Therefore, the values that are obtained from forcing are the ones that are used.
        p0_force=[.002, -100.]
        p0_fit=[.002, -100., thresh_inf]
        
        #TODO: THIS WOULD BE BETTER IF IT CALLED THE ACTUAL FUNCTION IN THE NEURON METHODS THAT WAY THEY WOULD HAVE TO BE THE SAME
        (popt_force, pcov_force)= curve_fit(exp_force_c, (time_previous_spike, thresh_inf), threshold, p0=p0_force, maxfev=100000)
        (popt_fit, pcov_fit)= curve_fit(exp_fit_c, time_previous_spike, threshold, p0=p0_fit, maxfev=100000)

        # viewing fit functions
        time_previous_spike.sort() #since time is not in order, making new time vector so that obtained fit curve can be plotted
        fit_force=exp_force_c((time_previous_spike, thresh_inf), popt_force[0], popt_force[1])
        fit_fit=exp_fit_c(time_previous_spike, popt_fit[0], popt_fit[1], popt_fit[2])
        if MAKE_PLOT:
            plt.plot(time_previous_spike, fit_force, 'r', lw=4, label="exp fit (force const to thesh first spike)\n  k=%.3g, amp=%.3g" % (popt_force[1], popt_force[0]))
            plt.plot(time_previous_spike, fit_fit, 'b', lw=4, label="exp fit (fit constant)\n  k=%.3g, amp=%.3g" % (popt_fit[1], popt_fit[0]))
            plt.legend()
            if SHOW_PLOT:        
                plt.show(block=False)
    
            if PUBLICATION_PLOT:       
                plt.figure(figsize=[14, 8])
                ax1=plt.subplot2grid((2, 2), (0,0))
                ax2=plt.subplot2grid((2, 2), (1,0))
                ax3=plt.subplot2grid((2, 2), (0,1), rowspan=2)
                for k in range(0, len(multi_SS_v)):
                    thresh=[multi_SS_v[k][j] for j in spike_ind[k]]
    
                    ax1.plot(np.arange(0, len(multi_SS_i[k]))*dt, multi_SS_i[k]*1.e12, lw=2)
                    ax1.set_ylabel('current (pA)', fontsize=16)
                    ax1.set_xlim([2., 2.12])
                    ax1.set_title('Triple Short Square', fontsize=20)
    
                    ax2.plot(np.arange(0, len(multi_SS_v[k]))*dt, multi_SS_v[k]*1.e3, lw=2)
                    ax2.plot(spike_ind[k]*dt, np.array(thresh)*1.e3, '.k', ms=16)
                    ax2.set_ylabel('voltage (mV)', fontsize=16)
                    ax2.set_xlabel('time (s)', fontsize=16)
                    ax2.set_xlim([2., 2.12])
                    
    
                ax3.plot(time_previous_spike, np.array(threshold)*1.e3, '.k', ms=16)
                ax3.set_ylabel('threshold (mV)', fontsize=16)
                ax3.set_xlabel('time since last spike (s)', fontsize=16)
                ax3.set_title('Spiking component of threshold', fontsize=20)
                ax3.plot(time_previous_spike, fit_force*1.e3, 'r', lw=4, label="exp fit: k=%.3g, amp=%.3g" % (popt_force[1], popt_force[0]))
                ax3.legend() 
                plt.tight_layout()
                plt.show()
        
        const_to_add_to_thresh_for_reset=popt_force[0]
        #REDICULOUS: this decay constant was originally forced to be positive and then it is negated everywhere
        #it is utalized elsewhere in the code.  For most neurons b_spike will be negative as the threshold decays.  
        #However that is not necessarily true--so using the abs would ignore those special neurons.  However, removing
        # abs would create problems for the rest of the code that negates this value, therefore I am negating the value 
        # here so that positive values will work correctly in the code for the special neurons.
        #this would be zero 
        #decay_const=abs(popt_force[1])
        decay_const=-popt_force[1]
        
        if decay_const <0:
            logging.CRITICAL('This neuron has an increasing decay value for the spike component of the threshold')
        if const_to_add_to_thresh_for_reset<0:
            logging.CRITICAL('This neuron has a negative amplitude for the spike component of the threshold') 
        
    except Exception, e:
        logging.error(e.message)
        const_to_add_to_thresh_for_reset=None 
        decay_const=None
        
    return const_to_add_to_thresh_for_reset, decay_const, thresh_inf

def err_fix_th(x, voltage, El, spike_cut_length, spikeInd, thi, dt, a_spike, b_spike):
    '''Based on calc_full_err_fixth module in test_fmin_fixth_ab.py created by Ram Iyer
    x[0]=a_spike input, x[1] is b_spike from multiblip'''
    #TODO: figure out how this works.
#    print "x, El, spike_cut_length, spikeInd, thi, dt, a_spike, b_spike"
#    print x, El, spike_cut_length, spikeInd, thi, dt, a_spike, b_spike   
    
    mba=a_spike
    mbb=b_spike
    t0=spikeInd
    vL = El
    fi = []                          #list to store sum of square errors for every pair of spikes
    for kk in range(len(t0)-1):      #loop over all (n-1) spikes in data
        sind = t0[kk]+int(spike_cut_length)
        eind = t0[kk+1]
        v=voltage[sind:eind-1]
#         offset = mba*np.exp(-mbb*dt*(t0[kk+1]-t0[kk])) #this offset is computed from your multi-blip data
#         theta2 = voltage[t0[kk+1]]-offset
        #######################################################################
        ##THIS IS THE PART THAT HAS BEEN MODIFIED FROM PREVIOUS CODE
        offset_sum = 0
        for jj in range(kk+1):
            offset = mba*np.exp(-mbb*dt*(t0[kk+1]-t0[jj])) #this offset is computed from your multi-blip data
            offset_sum = offset_sum + offset
                      
        theta2 = voltage[t0[kk+1]]-offset_sum  #threshold after subtracting spiking component
        #######################################################################
        theta1 = voltage[t0[kk]]*np.exp(-x[1]*dt*(eind-sind))
        delt = len(v)*dt
        tvec = np.arange(0,delt,dt)
        tvec = tvec[0:len(v)]
        
        lhs = theta2-theta1
        rhs2 = thi*(1-np.exp(-x[1]*dt*(eind-sind)))
        rhs1 = x[0]*np.exp(-x[1]*tvec[-1])*np.sum(dt*(v-vL)*np.exp(x[1]*tvec))
        rhs = rhs1+rhs2
        
        err = (lhs-rhs)**2
        if ~np.isnan(err):
            fi.append(err)
            
    F = np.sum(fi)
    
    #Print EVOLUTION OF ERROR FOR CONSISTENCY CHECK
    #print F*1e6
    
    return F

def find_multiblip_spikes(multi_SS_i, multi_SS_v, dt):
    '''artifacts caused by turning stimulus on and off created artifacts that 
    created problems for the standard spike detection algorithm.  Several alterations 
    were made so that the algorithm would detect spikes appropriately. Please see multiblip
    spike cutting documentation for more information on how the code differs from the SDK 
    version and what was needed to solve specific issues.
    input:
    multi_SS_i: list of arrays
        each array corresponds to the current stimulation of one triblip stimulus
    multi_SS_v: list of arrays
        each array corresponds to the voltage trace of triblip simulus
    dt: float 
    '''
    artifact_ave_window_time_s=0.0003 #
    window_indicies_to_ave_len=int(artifact_ave_window_time_s/dt)
    out_spk_idxs_list=[]
    for current, voltage in zip(multi_SS_i, multi_SS_v):
        #--Find the beginning and end of stimuli so that we can average the voltage traces at that time
        up_blip_index=np.where(np.diff(np.greater_equal(current, 1e-10).astype(int)) == 1)[0]#Note 1e-10 is larger than test pulse so it will not pick up test pulse
        down_blip_index=np.where(np.diff(np.less_equal(current, 1e-10).astype(int)) == 1)[0]
        potential_artifact_indexes=np.sort(np.append(up_blip_index, down_blip_index))
        artifact_removed_voltage=copy.deepcopy(voltage)
        window_boarders_index=[]
        #--remove artifacts 
        for index in potential_artifact_indexes:
            smooth_window=range(index,index+window_indicies_to_ave_len+1)
            
            # interpolate in the smoothing window
            blah=interp1d([smooth_window[0], smooth_window[-1]], [artifact_removed_voltage[smooth_window[0]], artifact_removed_voltage[smooth_window[-1]]])
            artifact_removed_voltage[smooth_window]=blah(smooth_window)
            
            #windows boarders are just for plotting
#            window_boarders_index.append(smooth_window[0])
#            window_boarders_index.append(smooth_window[-1])
#        plt.figure()
#        plt.plot(voltage, 'b', lw=4)
#        plt.plot(artifact_removed_voltage, 'r', lw=2)
#        plt.plot(window_boarders_index, artifact_removed_voltage[window_boarders_index], '|g', ms=10)
#        plt.xlim([40400, 41000])
#        plt.show()
#        
#        t = np.arange(0, len()) * dt

        # keeping the smooth_v convention of the SDK find spike code.  However in the SDK code 
        # this is used to name data potentially smoothed by a bessel filter
        smooth_v = artifact_removed_voltage
        dv = np.diff(smooth_v)    
        dvdt = dv / dt
        dvv = np.diff(dvdt)
                
        v=smooth_v[:-1]  #truncating the end of v so it has the same dimensions as dvdt for time plotting

        spikes = []
        out_spk_idxs = []
        
        peaks=get_peaks(v) # find potential spikes by finding peak over zero mv
      
        # Etay defines spike as time of threshold crossing.  Threshold is defined as the time at which dvdt is some percent of maximum threshold.
        # TODO: figure out how maximum threshold is defined in original code so I can say why I don't use it
        for spk_n, peak_idx in enumerate(peaks):
            #---------find spike peak----------------------------
            spk = {}
    
            spk["peak_idx"] = peak_idx 
            upstroke_idx = np.argmax(dvdt[peak_idx-int(.001/dt):peak_idx]) + peak_idx-int(.001/dt)  
            spk["upstroke"] = dvdt[upstroke_idx]
            spk["upstroke_idx"] = upstroke_idx
            spk["upstroke_v"] = v[upstroke_idx]
                
            # Define threshold where dvdt = 5% * max upstroke
            dvdt_thr_target = THRESH_PCT_MULTIBLIP * spk["upstroke"]
            #print 'spk[upstroke]', spk["upstroke"], 'dvdt_thr_target', dvdt_thr_target
            prev_idx = peak_idx-int(.0035/dt)
            #check to make sure prev_idx is not before or in a window where the stimulus blip comes on because 
            #it will errorniously trip the threshold dvdt
            for index in up_blip_index:
                if prev_idx<=index+int(.0005/dt) and prev_idx>= index-int(.0035/dt):
                    prev_idx=index+int(.0005/dt)
                    
            mean_dvv= [np.mean(dvv[pv-2:pv+3]) for pv in range(prev_idx,upstroke_idx)] #makes sure dv2/dt2 isnt spuriously going down by averaging 5 points
            find_thresh_idxs = np.where(np.logical_and(dvdt[prev_idx:upstroke_idx] >= dvdt_thr_target,  np.greater(mean_dvv,0)))[0]  
               
            if len(find_thresh_idxs) < 1: # Can't find a good threshold value - probably a bad simulation case
                # Fall back to the upstroke value
                threshold_idx = upstroke_idx
            else:
                threshold_idx = find_thresh_idxs[0] + prev_idx
                    
            spk["threshold_idx"] = threshold_idx
            spk["threshold_v"] = v[threshold_idx]
                
            # Check for things that are probably not spikes:
  
            # if the "spike" is less than 2 mV from threshold to peak, don't count it
            if v[peak_idx] - v[threshold_idx] < 0.002:  
                print "\tnot counting spike is closer to peak than 2 mV"
                continue
    
            #NOTE: because threshold doesnt decay to zero in the multiblip this doesnt get rid of the situation that usually is only in the first spike of a stimulus
            # if the spike is less the -30mV, don't count it
            if v[peak_idx] < -0.04:
                print "\tnot counting spike: peak is too small"
                continue
    
            spikes.append(spk)
    
    #----figure out if I should still find a global threshold and then do it all again
    #    # find global threshold which is an average of the individual thresholds
    #    if len(spikes) > 0:
    #        dvdt_thr_target = np.array([spk["upstroke"] for spk in spikes]).mean() * THRESH_PCT_MULTIBLIP
    #    else: # if there weren't any spikes, move along
    #        return np.array([])
        
            out_spk_idxs.append(spk["threshold_idx"])
        
        out_spk_idxs_list.append(np.array(out_spk_idxs))
        
#        time_vector=np.arange(len(v))*dt
#        plt.figure()
#    #    plt.subplot(3,1,1)
#    #    plt.plot(time_vector, ddv)
#    #    plt.plot(time_vector[out_spk_idxs], ddv[out_spk_idxs], '.r', ms=16)
#    #    plt.xlim([40300, 42000])
#    #    plt.ylabel('ddv')
#        plt.subplot(2,1,1)
#        plt.plot(time_vector, dvdt)
#        plt.plot(time_vector[out_spk_idxs], dvdt[out_spk_idxs], '.r', ms=16)
#        plt.plot(time_vector[potential_artifact_indexes], dvdt[potential_artifact_indexes], 'b|', ms=24, lw=4)
#        plt.xlim([40300*dt, 42000*dt])
#        plt.ylabel('dvdt')
#        plt.subplot(2,1,2)
#        plt.plot(time_vector, v)
#        plt.plot(time_vector[out_spk_idxs], v[out_spk_idxs], 'r.', ms=16, label='threshold')
#        plt.xlim([40300*dt, 42000*dt])
#        plt.ylabel('voltage (V)')
#        plt.plot(time_vector[peaks], v[peaks], '.g', ms=16, label='peaks')
#        plt.plot(time_vector[[spikes[ii]['upstroke_idx'] for ii in range(len(spikes))]], [spikes[ii]['upstroke_v'] for ii in range(len(spikes))], '.c', ms=16, label = 'max upstroke')
#        plt.plot(time_vector[potential_artifact_indexes], v[potential_artifact_indexes], 'b|', ms=24, lw=4)
#        plt.legend()
#        plt.show()
        
    return out_spk_idxs_list

def get_peaks(voltage, aboveValue=0):
    '''This function was written by Corinne Teeter and calculates the action potential peaks of a voltage equation"
    inputs
        voltage: numpy array of voltages
        aboveValue: scalar voltage value over which voltage is considered a spike.
    outputs:    
        peakInd: array of indicies of peaks'''
    VshiftR=np.concatenate(([0], voltage[0:voltage.size-2]))
    VshiftL=voltage[1:voltage.size]
    IndShiftR=np.where(voltage[0:voltage.size-1]>VshiftR)
    IndShiftL=np.where(voltage[0:voltage.size-1]>VshiftL)
    greatThanThresh=np.where(voltage>aboveValue) #finds indicies greater than the value provided
    peakInd=np.intersect1d(np.intersect1d(IndShiftL[0], IndShiftR[0]), greatThanThresh[0]) #find the indicies of the peak
    return peakInd

def exp_force_c((t, const), a1, k1):
    return a1*(np.exp(k1*t))+const

def exp_fit_c(t, a1, k1, const):
    return a1*(np.exp(k1*t))+const

