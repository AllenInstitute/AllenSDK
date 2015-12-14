import numpy as np

def calc_a_b_from_multiblip(multi_SS, dt, MAKE_PLOT=False, SHOW_PLOT=False, BLOCK=False, PUBLICATION_PLOT=False):
    '''In the multiblip there are problems with artifacts when the stimulus turns on and off 
    creating problems detecting spikes.
    '''    
    def exp_force_c((t, const), a1, k1):
        return a1*(np.exp(k1*t))+const

    def exp_fit_c(t, a1, k1, const):
        return a1*(np.exp(k1*t))+const

    multi_SS_v=multi_SS['voltage']
    multi_SS_i=multi_SS['current']

    spike_ind = find_spikes.find_multiblip_spikes(multi_SS_i, multi_SS_v, dt)
    # eliminate spurious spikes that may exist
    spike_lt=[np.where(SI<int(2.02/dt))[0] for SI in spike_ind]
    if len(np.concatenate(spike_lt))>0:
        warnings.warn('there is a spike before the stimulus in the multiblip')
    spike_ind=[np.delete(SI,ind)for SI, ind in zip(spike_ind, spike_lt)]
    spike_gt=[np.where(SI>int(2.1/dt))[0] for SI in spike_ind]
    if len(np.concatenate(spike_gt))>0:
        warnings.warn('there is a spike after the stimulus in the multiblip')    
    spike_ind=[np.delete(SI,ind)for SI, ind in zip(spike_ind, spike_gt)]
    
    #these are what I want to be final
    time_previous_spike=[]
    threshold=[]
    thresh_first_spike=[]  #will set constant to this
    
    if MAKE_PLOT:
        plt.figure()
    for k in range(0, len(multi_SS_v)):
        #voltage at all spikes in multiblip data
        thresh=[multi_SS_v[k][j] for j in spike_ind[k]]

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
        plt.show(block=BLOCK)

    #put numbers in one vector5
    thresh_inf=np.mean(thresh_first_spike)  #note this threshold infinity isnt the one coming from single blip
    print "average threshold of first spike",  thresh_inf
    try: #this try here because sometimes even though have the traces there isnt more than one trace with two spikes
        threshold=np.concatenate(threshold)
        time_previous_spike=np.concatenate(time_previous_spike)  #note that this will have nans in it
        
    #    for thr, times in zip(threshold, time_previous_spike):
    #        print thr
    #        print times
    #        if len(thr)!=len(times):
    #            print "not equal", 
    
        if MAKE_PLOT:
            plt.figure()
            plt.plot(time_previous_spike, threshold, '.k', ms=16)
            plt.ylabel('threshold (mV)')
            plt.xlabel('time since last spike (s)')
    
        
        p0_force=[.002, -100.]
        p0_fit=[.002, -100., thresh_inf]
    #    guess=exp_force_c(time_previous_spike, p0_force[0], p0_force[1])
        (popt_force, pcov_force)= curve_fit(exp_force_c, (time_previous_spike, thresh_inf), threshold, p0=p0_force, maxfev=100000)
        (popt_fit, pcov_fit)= curve_fit(exp_fit_c, time_previous_spike, threshold, p0=p0_fit, maxfev=100000)
        print 'popt_force', popt_force
    #    print 'popt_fit', popt_fit
        #since time is not in order lets make new time vector
        time_previous_spike.sort()
        fit_force=exp_force_c((time_previous_spike, thresh_inf), popt_force[0], popt_force[1])
        fit_fit=exp_fit_c(time_previous_spike, popt_fit[0], popt_fit[1], popt_fit[2])
    #    plt.plot(time_previous_spike, guess, label='guess')
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
    
    #                plt.subplot(2,1,1)
                    ax1.plot(np.arange(0, len(multi_SS_i[k]))*dt, multi_SS_i[k]*1.e12, lw=2)
                    ax1.set_ylabel('current (pA)', fontsize=16)
                    ax1.set_xlim([2., 2.12])
                    ax1.set_title('Triple Short Square', fontsize=20)
    
    #                plt.subplot(2,1,2)
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
                
        #TODO: Corinne abs is put in hastily make sure this is ok
        const_to_add_to_thresh_for_reset=abs(popt_force[0]) 
        b=abs(popt_force[1])
    except:
        const_to_add_to_thresh_for_reset=None 
        b=None
        
    return const_to_add_to_thresh_for_reset, b, thresh_inf

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
