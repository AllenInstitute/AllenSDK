import numpy as np

def R2R_subthresh_nonlinearity(rheo_current_list, rheo_voltage_list, res, cap, El, El_R2R_list, dt, 
                               MAKE_PLOT=False, SHOW_PLOT=False, BLOCK=False):
    '''Estimates how resistance is dependent on voltage before a spike.  Fits two lines to resistance 
    and voltage output of epochs of noise. 
    inputs:
        rheo_current_list: list of stimulation inputs for each sweep repeat with test pulse removed
        rheo_voltage_list: list of voltage recorded for each sweep repeat_with test pulse removed
        res, cap,  and El: floats previously estimated (usually use values from least squares of subthreshold noise) 
        El_R2R_list: list of resting potentials. Each value corresponds to one sweep repeat
        dt: time step size of the data
    outputs:
        line_param_RV: parameters of two lines fit to voltage and resistance [
        line_param_ElV: parameters of two lines fit to voltage and resistance
    '''
    
    #TODO: talk to Ram and Stefan about weather we should be using two different El's in this function

    if MAKE_PLOT:
        plt.figure(figsize = (12,15))
        colors=['r', 'g', 'b', 'c']

    line_param_RV_list=[]
    line_param_ElV_list=[]
    all_sweep_res=np.array([])
    all_sweep_v=np.array([])
    all_sweep_El=np.array([])

    for ss in range(0, len(rheo_current_list)):  #loop over stimulus repeats
        rheo_current=np.array(rheo_current_list[ss]) 
        rheo_voltage=np.array(rheo_voltage_list[ss])  
        El_R2R=El_R2R_list[ss]
        rheo_first_spike_ind=find_spikes.find_spikes([rheo_voltage], 'dvdt_v2', dt)[0]-20                
        #Choose start and end timepoints for ramp
        tst=find_stimulus(rheo_current)[0]
        if rheo_first_spike_ind.any():
            tend=rheo_first_spike_ind[0]-int(.005/dt)  #choose 5 ms arbitrarily
        else:
            tend=len(rheo_current)
        noise_period_len=int(1./dt) #chunking of the data (1 s, because that is the length of each noise seed)
        
        #define subthreshold section of ramp to rheo stimulus.                
        v = rheo_voltage[tst:tend-1]-El_R2R  #look to see how different El's are if dont subtract El-R2R (if dont need it, don't need to think about which el to use)
        i = rheo_current[tst:tend-1]
        
        #Specify intervals for individual calculations
        start_index_of_last_subthresh_noise_period = noise_period_len*int(len(v)/noise_period_len) + noise_period_len
        intervals = np.arange(0,start_index_of_last_subthresh_noise_period,noise_period_len) #starting indicies noise periods containing subthreshold voltage
        
        #segment data in intervals
        segmented_voltage_list=[]
        segmented_current_list=[]
        segmented_v_mean_list=[]
        for start_index_of_interval in intervals:
            segmented_voltage_list.append(v[start_index_of_interval:start_index_of_interval+noise_period_len])
            segmented_current_list.append(i[start_index_of_interval:start_index_of_interval+noise_period_len])
            segmented_v_mean_list.append(np.mean(segmented_voltage_list[-1]))

        #Run leqst squares for every interval to find R, El
        (R_lssq_list, El_lssq_list)=least_squares_simple_circuit_fit_REl(segmented_voltage_list, segmented_current_list, cap, dt) #note a list will come out of this function but it is only one number   
#        res_array_MOhm=np.array(R_lssq_list)*1e-6
#        cap_array_pF=np.array(C_lssq_list)*1e12
#        tau_array_ms=np.array(R_lssq_list)*np.array(C_lssq_list)*1e3
#        El_array_mV=np.array(El_lssq_list)*1e3
#        mean_v_array_mV=np.array(segmented_v_mean_list)*1e3
        
        res_array=np.array(R_lssq_list)
        tau_array=np.array(R_lssq_list)*cap
        El_array=np.array(El_lssq_list)
        mean_v_array=np.array(segmented_v_mean_list)
        
        all_sweep_res=np.append(all_sweep_res, res_array)
        all_sweep_v=np.append(all_sweep_v, mean_v_array)
        all_sweep_El=np.append(all_sweep_El, El_array)
        
        if len(mean_v_array) >= 3: #need at lest 3 points to perform a fit
            # Fit piecewise linear function to R(V)
            p0_0=np.mean(res_array[0:1])
            m=(res_array[-1]-res_array[-2])/(mean_v_array[-1]-mean_v_array[-2])
            b=res_array[-1]-m*mean_v_array[-1]
             
            print 'in Piecewise non linearity: initial conditions are ', p0_0, m, b 
            line_param_RV, cov = curve_fit(max_of_line_and_const, mean_v_array, res_array, p0=[p0_0, m, b], maxfev = 1000000) #fit piecewise linear function
            crossover = (line_param_RV[0] - line_param_RV[2]) / (line_param_RV[1])   #voltage where the two lines crossover   
            print 'line_param_RV', line_param_RV
            print 'crossover', crossover   
     
            # Fit piecewise linear function to El(V)
            line_param_ElV, cov2 = curve_fit(min_of_line_and_zero, mean_v_array, El_array, p0=[-10,  0], maxfev = 1000000)
#            crossover2 = (line_param_ElV[0] - line_param_ElV[2]) / (line_param_ElV[1])
            crossover3 = (line_param_ElV[1]) / (line_param_ElV[0])
            print 'line_param_ElV', line_param_ElV
            print 'crossover3', crossover3   
    
            if MAKE_PLOT:
                plt.subplot(3,1, 1)
                plt.plot(np.arange(len(v))*dt*1000., v*1e3, color=colors[ss], label='repeat'+str(ss))
                for line in intervals:
                    plt.axvline(x=line*dt*1000, color='k', ls='--')
#                plt.plot(np.array(intervals)*dt*1000, np.array(v[intervals])*1e3, '|k', ms=24, mew=4)
                plt.xlabel('time (ms)')
                plt.ylabel('voltage (mV)')
                plt.title("Subthreshold section of ramp to rheo stimulus")
                
                plt.subplot(3,2,3)
                plt.plot(mean_v_array*1e3, res_array/1e6, 'o', color=colors[ss])
                plt.plot(mean_v_array*1e3, max_of_line_and_const(mean_v_array, *line_param_RV)/1e6, '--', color=colors[ss])
                plt.title('fit of resistance')
                plt.xlabel('V (mV)')
                plt.ylabel('resistance (MOhms)')
                
                plt.subplot(3,2,4)
                plt.plot(mean_v_array*1e3, El_array*1e3, 'o', color=colors[ss])
                plt.plot(mean_v_array*1e3, min_of_line_and_zero(mean_v_array, *line_param_ElV)*1e3, '--', color=colors[ss])
                plt.title('fit of resting potential')
                plt.xlabel('V (mV)')
                plt.ylabel('El (mV)')
            
        else:    
            # Fit piecewise linear function to R(V)
            line_param_RV=[]
            crossover=[]  
            print 'line_param_RV', line_param_RV
            print 'crossover', crossover   
     
            # Fit piecewise linear function to El(V)
            line_param_ElV=[]            
#            crossover2 = []
            crossover3 = []
            print 'line_param_ElV', line_param_ElV
            print 'crossover3', crossover3   
    
            if MAKE_PLOT:
                plt.subplot(3,1, 1)
                plt.plot(np.arange(len(v))*dt*1000., v, '--', color=colors[ss], label='repeat'+str(ss)+' (not used in analysis)')
                plt.xlabel('time (ms)')
                plt.ylabel('voltage (V)')
                plt.title("Subthreshold section of ramp to rheo stimulus")

                
        line_param_RV_list.append(line_param_RV)
        line_param_ElV_list.append(line_param_ElV)
    
    #--Using data from all sweeps here 
    # Fit piecewise linear function to R(V)
    
    if len(all_sweep_v)>=3:
        p0_0=np.mean(all_sweep_res[0:1])
        m=(all_sweep_res[-1]-all_sweep_res[-2])/(all_sweep_v[-1]-all_sweep_v[-2])
        b=all_sweep_res[-1]-m*all_sweep_v[-1]
         
        print 'in Piecewise non linearity: initial conditions are ', p0_0, m, b 
        line_param_RV, cov = curve_fit(max_of_line_and_const, all_sweep_v, all_sweep_res, p0=[p0_0, m, b], maxfev = 1000000) #fit piecewise linear function
        crossover = (line_param_RV[0] - line_param_RV[2]) / (line_param_RV[1])   #voltage where the two lines crossover   
        print 'line_param_RV', line_param_RV
        print 'crossover', crossover   
    
        # Fit piecewise linear function to El(V)
        line_param_ElV, cov2 = curve_fit(min_of_line_and_zero, all_sweep_v, all_sweep_El, p0=[-10,  0], maxfev = 1000000)
#        crossover2 = (line_param_ElV[0] - line_param_ElV[2]) / (line_param_ElV[1]) #non zero const line
        crossover3 = (line_param_ElV[1]) / (line_param_ElV[0])
        print 'line_param_ElV', line_param_ElV
        print 'crossover3', crossover3   
    
        if MAKE_PLOT:
            sorted_v=np.sort(all_sweep_v)
            plt.subplot(3,2,5)
            plt.plot(all_sweep_v*1e3, all_sweep_res/1e6, 'o', color=colors[ss])
            plt.plot(sorted_v*1e3, max_of_line_and_const(sorted_v, *line_param_RV)/1e6, '--', color=colors[ss])
            plt.title('fit of resistance (all sweeps)')
            plt.xlabel('V (mV)')
            plt.ylabel('resistance (MOhms)')
            
            plt.subplot(3,2,6)
            plt.plot(all_sweep_v*1e3, all_sweep_El*1e3, 'o', color=colors[ss])
            plt.plot(sorted_v*1e3, min_of_line_and_zero(sorted_v, *line_param_ElV)*1e3, '--', color=colors[ss])
            plt.title('fit of resting potential (all sweeps)')
            plt.xlabel('V (mV)')
            plt.ylabel('El (mV)')
    else:
        if MAKE_PLOT:
            sorted_v=np.sort(all_sweep_v)
            plt.subplot(3,2,5)
            plt.plot(all_sweep_v*1e3, all_sweep_res/1e6, 'o', color=colors[ss])
            plt.title('fit of resistance (all sweeps)')
            plt.xlabel('V (mV)')
            plt.ylabel('resistance (MOhms)')
            
            plt.subplot(3,2,6)
            plt.plot(all_sweep_v*1e3, all_sweep_El*1e3, 'o', color=colors[ss])
            plt.title('fit of resting potential (all sweeps)')
            plt.xlabel('V (mV)')
            plt.ylabel('El (mV)')

    if MAKE_PLOT:
        plt.legend()
        plt.tight_layout()
    if SHOW_PLOT:
        plt.show(block = BLOCK)
                      
    return line_param_RV_list, line_param_ElV_list, line_param_RV, line_param_ElV
