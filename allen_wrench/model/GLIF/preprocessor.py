import load_data
import numpy as np
import AIC

from scipy import signal
import warnings
from scipy import stats
from scipy.io import savemat
from scipy.optimize import curve_fit
import sys
import statsmodels.api as sm

import logging
logger = logging.getLogger()
DEBUG_MODE = logger.isEnabledFor(logging.DEBUG)
if DEBUG_MODE:
    import matplotlib.pylab as plt
    import plotting
    import time


class GLIFPreprocessor(object):
    def __init__(self, neuron_config, stimulus_file_name, sweep_properties, optional_methods=None):
        self.neuron_config = neuron_config
        self.stimulus_file_name = stimulus_file_name
        self.sweep_properties = sweep_properties

        self.optional_methods = optional_methods
        if self.optional_methods is None:
            self.optional_methods = {}

        self.optimize_data = None
        self.superthreshold_blip_data = None
        self.subthreshold_blip_data = None

        self.spike_ind_list = None
        self.interpolated_spike_time_target_list = None
        self.grid_spike_time_target_list = None
        self.target_spike_mask = None

    @staticmethod
    def load_stimulus_basic(file_name, sweeps, dt=None, cut=0, bessel_filter=False):
        assert sweeps, Exception("Error: no sweeps to load")
        
        sweep_data = load_data.load_sweeps(file_name, sweeps)

        num_sweeps = len(sweeps)

        if cut > 0:
            for i in xrange(num_sweeps):
                sweep_data['voltage'][i] = sweep_data['voltage'][i][cut:]
                sweep_data['current'][i] = sweep_data['current'][i][cut:]

        # if a bessel filter is requested with no params, supply a default
        if bessel_filter is True:
            bessel_filter = { 'N': 4, 'Wn': .01 }
        
        if bessel_filter:
            b, a = signal.bessel(bessel_filter['N'], bessel_filter['Wn'])
            for i in xrange(num_sweeps):
                d[i]['voltage'] = signal.filtfilt(b, a, d[i]['voltage'], axis=0)

        # if a specific dt is request, subsample as necessary
        if dt is not None:
            for i in xrange(num_sweeps):
                # check the current dt of the sweep and modify if it's different from the requested dt
                if sweep_data['dt'][i] != dt:
                    sweep_data['voltage'][i] = subsample_data(sweep_data['voltage'][i], 'mean', sweep_data['dt'][i], dt)
                    sweep_data['current'][i] = subsample_data(sweep_data['current'][i], 'mean', sweep_data['dt'][i], dt)
                    sweep_data['start_idx'][i] = int(sweep_data['start_idx'][i] / (dt / sweep_data['dt'][i]))
                    sweep_data['dt'][i] = dt
        
        return sweep_data

    def load_stimulus(self, file_name, sweeps):
        subsample_config = self.optional_methods.get("subsample", {})
        dt = subsample_config.get("time_step", None)

        cut_config = self.optional_methods.get("cut", {})
        cut = cut_config.get("front", 0)

        bessel_filter = self.optional_methods.get("bessel_filter", False)
        
#------------------------------------------------------------------------------------------------
#------------------for now I am commenting this out because I think this is now implemented------
#------------------automatically comming from MIES--maybe there should be a check to make--------
#------------------sure it is happening.  To do this we would need that list of indices --------
#------------------to make sure the current is zero there----------------------------------------
#        if self.optional_methods.get('cut_extra_current', False):
#            assert id_of_silence_list is not None, Exception("cut_extra_current requires an id_of_silence_list")
#
#            extra_current_ids = [ id_of_silence_list[ii] for ii in sweeps ]
#            cut_extra_current(self.stim_list, self.bio_list, extra_current_ids)  
   
        sweep_data = self.load_stimulus_basic(file_name, sweeps, dt, cut, bessel_filter)

        zero_out_el = self.optional_methods.get("zero_out_el_via_inj_current", False)
                
        if zero_out_el:
            for i in xrange(len(sweeps)):
                zero_out_el_via_inj_current(sweep_data['current'][i], self.El_reference, self.sweep_properties[i]['resting_potential'], self.neuron_config['R_input'])

        return sweep_data

        
    def preprocess_stimulus(self, optimize_sweeps, superthreshold_blip_sweeps, subthreshold_blip_sweeps, ramp_sweeps, all_noise_sweeps, multi_blip_sweeps, spike_determination_method='threshold'):
        
        #TODO:  instead of just taking zeroth sweep there should be should way to take the first passed sweep
        
        # reference resting potential comes from the ramp
        sec_before_thresh_to_look_at_spikes=.002 #used in spike cutting
        self.El_reference = self.sweep_properties[ramp_sweeps[0]]['resting_potential']  * 1e-3
        if self.El_reference>-.03 or self.El_reference<-.1:
            print "the resting potential is :", self.El_reference
            raise Exception("Resting potential is not in the correct range")
        self.neuron_config['El'] = 0  # AT THE MOMENT EVERYTHING IS BEING CALCULATED SHIFTING EL TO ZERO

        # --load the sweeps that will be run in the model optimizer
        self.optimize_data = self.load_stimulus(self.stimulus_file_name, optimize_sweeps)
        dt = self.optimize_data['dt'][0]
        self.neuron_config['dt'] = dt

        # calculate input resistance from the slowest ramp that producess a spike. for now only using the first ramp
        ramp_data = self.load_stimulus(self.stimulus_file_name, ramp_sweeps)
        
        R_via_begining_of_ramp = calculate_input_resistance_via_ramp(ramp_data['voltage'][0][ramp_data['start_idx'][0]:] - self.El_reference, 
                                                                           ramp_data['current'][0][ramp_data['start_idx'][0]:], self.neuron_config['dt'])                    
        print 'Resistance via begining of ramp: ', R_via_begining_of_ramp
        
        #calculate resistance via average subthreshold noise
        noise_data=self.load_stimulus(self.stimulus_file_name,all_noise_sweeps)
        subthresh_noise_voltage=noise_data['voltage'][0][0:int(6./dt)]
        subthresh_noise_current=noise_data['current'][0][0:int(6./dt)]
        noise_El=self.sweep_properties[all_noise_sweeps[0]]['resting_potential']*1e-3
        
        R_via_noise=calculate_input_R_via_subthresh_noise(subthresh_noise_voltage, 
                                                          subthresh_noise_current, noise_El, noise_data['start_idx'][0], dt)
        print 'resistance via noise average', R_via_noise
        
        #TODO put delta somewhere else
        (R_via_ave20to40percent_ramp, deltaV_via_ave20to40percent_ramp)=calc_input_R_and_extrapolatedV_via_ramp(ramp_data['voltage'][0], ramp_data['current'][0], 
                                                                                       ramp_data['start_idx'][0], dt, self.El_reference)

        print "Resistance averaging the resitance over 20 to 40 percent of ramp:", R_via_ave20to40percent_ramp
        print 'difference between threshold on ramp and threshold via extrapolated V', deltaV_via_ave20to40percent_ramp
        
        (R_glm, C_glm)=RC_via_subthresh_GLM(subthresh_noise_voltage, subthresh_noise_current, dt)
        print "R via GLM", R_glm*1e-6, "Mohms"
        print "C via GLM", C_glm*1e12, "pF"

        #compute R, C, and El via least squares
        (R_lssq, C_lssq, El_lssq)=R_via_subthresh_linRgress(subthresh_noise_voltage, subthresh_noise_current, dt)
        print "difference in resting potential calculations", self.El_reference-El_lssq

        self.neuron_config['R_input']=R_lssq
  
#        #---these are for saving for Ram
#        out={'I_stim':subthresh_noise_current, 'voltage':subthresh_noise_voltage, 'dt':dt}
#        savemat('/data/mat/RamIyer/IVSCC/subthresh_noise5', out)
#        sys.exit()

        # calculate capacitance from subthreshold blip data
        self.subthreshold_blip_data = self.load_stimulus(self.stimulus_file_name, subthreshold_blip_sweeps) 

#        #this approximation might not be doing so well because of capacitance compensation need to get v by fitting 2 exponentials
#        self.neuron_config['C'] = calculate_capacitance_via_subthreshold_blip(self.subthreshold_blip_data['voltage'][0][self.subthreshold_blip_data['start_idx'][0]:] - self.El_reference, 
#                                                                                self.subthreshold_blip_data['current'][0][self.subthreshold_blip_data['start_idx'][0]:], self.neuron_config['dt'])

        (cap_via_charge_dump, total_charge_dump)=calc_cap_via_fit_charge_dump(self.subthreshold_blip_data['voltage'][0][self.subthreshold_blip_data['start_idx'][0]:], 
                                     self.subthreshold_blip_data['current'][0][self.subthreshold_blip_data['start_idx'][0]:], 
                                     self.neuron_config['R_input'], dt, self.sweep_properties[subthreshold_blip_sweeps[0]]['resting_potential']*1e-3)
        print "capacitance via charge dump fit extrapolation", cap_via_charge_dump

        self.neuron_config['C']=C_lssq
        
        #calculate b constant and potential threshold reset
        muliti_SS = self.load_stimulus(self.stimulus_file_name, multi_blip_sweeps)
    
        (const_to_add_to_thresh_for_reset,b_value)=calc_a_b_from_muliblip(muliti_SS, dt)
        #TODO: 
        print "value added to previous threshold if method is V_plus_const", const_to_add_to_thresh_for_reset
        
        deltaV_ramp_and_used_R=calc_deltaV_via_specified_R_and_ramp(self.neuron_config['R_input'], ramp_data['voltage'][0], ramp_data['current'][0], 
                                                                                       ramp_data['start_idx'][0], dt, self.El_reference)
#        deltaV=0
        deltaV=deltaV_via_ave20to40percent_ramp
#        deltaV=deltaV_ramp_and_used_R
        
        # extract instantaneous threshold from superthreshold blip
        self.superthreshold_blip_data = self.load_stimulus(self.stimulus_file_name, superthreshold_blip_sweeps) 
        thresh_inf_via_Vmeasure = find_first_spike_voltage(self.superthreshold_blip_data['voltage'][0], dt) - self.El_reference     
        print "threshold infinity measured from blip", thresh_inf_via_Vmeasure
        print "threshold infinity measured from blip minus V difference of ramp thesh verusus extrapolated threhsold", thresh_inf_via_Vmeasure-deltaV
                    
        # calculate adaptive threshold from ramp
        #TODO: this is now repetitive and why the different load statements
        ramp_data = self.load_stimulus(self.stimulus_file_name, ramp_sweeps) 
        self.neuron_config['th_adapt'] = find_first_spike_voltage(ramp_data['voltage'][0], dt) - self.El_reference-deltaV
        thresh_adapted_read_from_ramp= find_first_spike_voltage(ramp_data['voltage'][0], dt) - self.El_reference

        #calculate threshold inf via R and c and charge dump
        thresh_inf_via_Q_and_C=total_charge_dump/self.neuron_config['C']
        self.neuron_config['th_inf']=thresh_inf_via_Q_and_C
        print "threshold infinity calculated with charge dump and calculated C", thresh_inf_via_Q_and_C

        a_value=b_value*(thresh_adapted_read_from_ramp-thresh_inf_via_Vmeasure)/(thresh_adapted_read_from_ramp-self.neuron_config['El'])
        if abs(a_value)<0.05*abs(b_value):
            warnings.warn('a is less than 5 percent of b.  Adjusting a to be 5 percent')
            if a_value<0.0:
                a_value=-.05*b_value
            elif a_value>=0.0:
                a_value=.05*b_value
        print 'a', a_value
        print 'b', b_value                                                                                                                                                    
        
        print 'delta V used to adjust thresholds', deltaV
        print 'Adapted threshold used', self.neuron_config['th_adapt']
        print 'Instantaneous threshold used', self.neuron_config['th_inf']
        print 'capacitance used', self.neuron_config['C']
        print 'resistance used', self.neuron_config['R_input']
      
        compute_spike_cut_length = 'spike_cut_length' not in self.neuron_config
        
        #---------------------------------------------------------------------------
        #--------------calculate reset rules----------------------------------------
        #--------------------------------------------------------------------------- 

        # configure the voltage reset method

        method_config = self.neuron_config['voltage_reset_method']
        if method_config.get('params', None) is None:
            if method_config['name'] == 'fixed':

                # --load all noise
                endPointMethod = 'timeAfterThresh'
                noise_data = self.load_stimulus(self.stimulus_file_name, all_noise_sweeps)

                (noise_v_spike_shape_list, noise_i_spike_shape_list, noise_zeroCrossInd, noise_thresholdInd, waveIndOfFirstSpikes, spikeFromWhichSweep) \
                    = align_and_cut_spikes(noise_data['voltage'], noise_data['current'], dt=dt, method=spike_determination_method, sec_look_before_spike=sec_before_thresh_to_look_at_spikes)
                ave_V_trace=np.mean(noise_v_spike_shape_list, axis=0)
                #TODO: CorinneT: all this hard coded index stuff is not good it will be dependent on the hard coded stuff in align_and_cut_spikes
                #TODO: CorinneT: a better algorithm might be the first deflection of the derivative (just incase the voltage goes up then down again
                min_v=np.min(ave_V_trace[.004/dt:.010/dt]) 
                if min_v==ave_V_trace[.010/dt-1]: #if the minimum is equal to the end of the spike then choose a time 4 ms after the spike threshold as min
                    min_v=ave_V_trace[.0065/dt] #this is ~4 ms after thereshold. TODO: Note this will be dependent on how spikes are cut: fix this  
                index_of_min=np.where(ave_V_trace==min_v)[0][0]
                if DEBUG_MODE:
                    plt.figure()
                    for i,v in zip(noise_i_spike_shape_list, noise_v_spike_shape_list):
                        plt.subplot(2,1,1)
                        plt.plot(np.arange(0, len(i))*dt, i)
                        plt.subplot(2,1,2)
                        plt.plot(np.arange(0, len(v))*dt, v)
                    plt.subplot(2,1,1)
                    plt.title('average spike shape for fixed voltage reset')
                    plt.subplot(2,1,2)
                    plt.plot(np.arange(0, len(ave_V_trace))*dt, ave_V_trace, '--k', lw=6, label='mean')
                    plt.plot(int(index_of_min)*dt, ave_V_trace[index_of_min], 'kx', ms=16, lw=6, label='min v')
                    plt.xlabel('time (s)')
                    plt.ylabel('voltage (V)')
                    plt.legend()
                    plt.show(block=False)

                method_config['params'] = {
                    'value': min_v-self.El_reference-deltaV
                }

                if compute_spike_cut_length:
                    self.neuron_config['spike_cut_length'] = index_of_min-int(sec_before_thresh_to_look_at_spikes/dt)
            elif method_config['name'] == 'LIF':
                #NOTE I AM ONLY DOING THIS HERE TO GET A VALUE FOR SPIKE CUTTING
                endPointMethod = 'timeAfterThresh'
                noise_data = self.load_stimulus(self.stimulus_file_name, all_noise_sweeps)

                (noise_v_spike_shape_list, noise_i_spike_shape_list, noise_zeroCrossInd, noise_thresholdInd, waveIndOfFirstSpikes, spikeFromWhichSweep) \
                    = align_and_cut_spikes(noise_data['voltage'], noise_data['current'], dt=dt, method=spike_determination_method, sec_look_before_spike=sec_before_thresh_to_look_at_spikes)
                ave_V_trace=np.mean(noise_v_spike_shape_list, axis=0)
                #TODO: CorinneT: all this hard coded index stuff is not good it will be dependent on the hard coded stuff in align_and_cut_spikes
                #TODO: CorinneT: a better algorithm might be the first deflection of the derivative (just incase the voltage goes up then down again
                min_v=np.min(ave_V_trace[.004/dt:.010/dt]) 
                if min_v==ave_V_trace[.010/dt-1]: #if the minimum is equal to the end of the spike then choose a time 4 ms after the spike threshold as min
                    min_v=ave_V_trace[.0065/dt] #this is ~4 ms after thereshold. TODO: Note this will be dependent on how spikes are cut: fix this  
                index_of_min=np.where(ave_V_trace==min_v)[0][0]
                method_config['params'] = {
                    'value': 0.0
                } 
                if compute_spike_cut_length:
                    warnings.warn('you are cutting spikes via minimum average spike trace')
                    self.neuron_config['spike_cut_length'] = index_of_min-int(sec_before_thresh_to_look_at_spikes/dt)   
            elif method_config['name'] == 'Vbefore':
                # --load all noise
                endPointMethod = 'timeAfterThresh'
                noise_data = self.load_stimulus(self.stimulus_file_name, all_noise_sweeps)

                #TODO: CorinneT put in other stimuli 
                all_voltage_list = noise_data['voltage']
                all_current_list = noise_data['current']

                (all_v_spike_shape_list, all_i_spike_shape_list, all_zeroCrossInd, all_thresholdInd, waveIndOfFirstSpikes, spikeFromWhichSweep) \
                    = align_and_cut_spikes(all_voltage_list, all_current_list, dt=dt, method=spike_determination_method, sec_look_before_spike=sec_before_thresh_to_look_at_spikes)
                
                #change frame of reference of voltage
                all_v_spike_shape_list=[shape-self.El_reference-deltaV for shape in all_v_spike_shape_list]
                                
                if DEBUG_MODE:
                    plt.title(self.stimulus_file_name + ' Spike cutting', fontsize=22)    
                    plt.show(block=False)   

                # --setting limits to find explained variance
                #TODO: CorinneT: more scary indicies
                expVarIndRangeAfterSpike = range(int(.003 / dt), int(.010 / dt))  # =[int(.003/dt)]  NOTE: THIS IS USED IN REFERENCE TO SPIKE TIME
                vectorIndex_of_max_explained_var = expVarIndRangeAfterSpike[0]  # this is just here for the title of the plot
                list_of_endPointArrays = []  # this should end up a list of numpy arrays where each numpy array contains the indices of the v_spike_shape_list that are a certain time after the threshold
                if endPointMethod is 'timeAfterThresh':
                    for ii in expVarIndRangeAfterSpike:
                        list_of_endPointArrays.append(np.array(all_thresholdInd) + ii)
                elif endPointMethod is 'timeAfterZeroCross':
                    for ii in expVarIndRangeAfterSpike:
                        list_of_endPointArrays.append(np.array(all_zeroCrossInd) + ii)
                        
                linRegress_error_4_each_time_end = []
                varData_4_each_time_end = []
                varModel_4_each_time_end = []
                chi2 = []
                nonBlip_v_spike_shape_list = all_v_spike_shape_list
                nonBlip_thresholdInd = all_thresholdInd
                xdata = np.array([v[nonBlip_thresholdInd[ii]] for ii, v in enumerate(nonBlip_v_spike_shape_list)])
                var_of_Vdata_beforeSpike = np.var(xdata)
                for jj, vectorOfIndAcrossWaves in enumerate(list_of_endPointArrays):  # these indices should be in terms of the spike waveforms
            #        print 'jj', jj
                    # TODO: Teeter get rid of the nonblipness
                    nonBlip_v_at_specificEndPoint = [nonBlip_v_spike_shape_list[ii][index] for ii, index in enumerate(vectorOfIndAcrossWaves)]  # this is calculating variance at certain time points
                    # --currently the model of voltage reset is a linear regression between voltage before the spike and the voltage after the spike but it could be more complicated (for example as a function of current) 
                    ydata = np.array(nonBlip_v_at_specificEndPoint)  # this is the voltage at the specified end point
                    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
                    ymodel = slope * xdata + intercept
                    residuals = ydata - ymodel
                    chi2.append(np.var(residuals))  # how well the model describes the data
                    linRegress_error_4_each_time_end.append(std_err)
                    varData_4_each_time_end.append(np.var(nonBlip_v_at_specificEndPoint))
                    varModel_4_each_time_end.append(np.var(ymodel))
                # --these will line up with how many arrays there are in the list
                vectorIndex_of_min_var = varData_4_each_time_end.index(min(varData_4_each_time_end))
                vectorIndex_of_min_stderr = linRegress_error_4_each_time_end.index(min(linRegress_error_4_each_time_end))
                vectorIndex_of_min_chi2 = chi2.index(min(chi2))
                # --components of calculating the explained variance
                var_of_Vdata_afterSpike = np.array(varData_4_each_time_end)
                var_of_Vmodel_afterSpike = np.array(varModel_4_each_time_end)
                # numerator=(var_of_Vdata_beforeSpike+var_of_Vdata_afterSpike-np.array(chi2))  this is wrong (found 10-27-14)
                # denominator=(var_of_Vdata_beforeSpike+var_of_Vdata_afterSpike) this is wrong (found 10-27-14)
                numerator = (var_of_Vdata_afterSpike + var_of_Vmodel_afterSpike - np.array(chi2))
                denominator = (var_of_Vdata_afterSpike + var_of_Vmodel_afterSpike)
                explained_var_at_each_set_of_endPoint = np.divide(numerator, denominator)
                explained_diffvar_at_each_set_of_endPoint = np.diff(explained_var_at_each_set_of_endPoint)
                explained_diff2var_at_each_set_of_endPoint = np.diff(explained_diffvar_at_each_set_of_endPoint)
                vectorIndex_of_max_explained_var = explained_var_at_each_set_of_endPoint.tolist().index(max(explained_var_at_each_set_of_endPoint))
                vectorIndex_of_max_diff_explained_var = explained_diffvar_at_each_set_of_endPoint.tolist().index(max(explained_diffvar_at_each_set_of_endPoint))
                vectorIndex_of_max_diff2_explained_var = explained_diff2var_at_each_set_of_endPoint.tolist().index(max(explained_diff2var_at_each_set_of_endPoint))          
                
                if DEBUG_MODE:
                    plt.figure(figsize=(20, 10))
                    plt.subplot(5, 1, 1)
                    plt.plot(np.array([float(jj) for jj in expVarIndRangeAfterSpike]) * dt, varData_4_each_time_end)
                    plt.plot(float(expVarIndRangeAfterSpike[vectorIndex_of_min_var]) * dt, varData_4_each_time_end[vectorIndex_of_min_var], 'xk', ms=16, lw=10)
                    plt.ylabel('variance')
            #        plt.title(cellTitle+', '+title_endPoint_type_string+'\nmin var='+ str(varData_4_each_time_end[vectorIndex_of_min_var])+', at t='+str(expVarIndRangeAfterSpike[vectorIndex_of_min_var]*dt))
#                    plt.title(cellTitle, fontsize=20)
                    plt.subplot(5, 1, 2)
                    plt.plot(np.array([float(jj) for jj in expVarIndRangeAfterSpike]) * dt * dt, chi2)
                    plt.plot(float(expVarIndRangeAfterSpike[vectorIndex_of_min_chi2]) * dt, chi2[vectorIndex_of_min_chi2], 'xk', ms=16, lw=10)
                    plt.ylabel('chi2')
                    plt.title('min chi2=' + str(chi2[vectorIndex_of_min_chi2]) + ', at t=' + str(expVarIndRangeAfterSpike[vectorIndex_of_min_chi2] * dt))
                    plt.subplot(5, 1, 3)
                    plt.plot(np.array([float(jj) for jj in expVarIndRangeAfterSpike]) * dt, explained_var_at_each_set_of_endPoint)
                    plt.plot(float(expVarIndRangeAfterSpike[vectorIndex_of_max_explained_var]) * dt, explained_var_at_each_set_of_endPoint[vectorIndex_of_max_explained_var], 'xk', ms=16, lw=10)
                    plt.ylabel('explained variance')
                    plt.title('max explained variance=' + str(explained_var_at_each_set_of_endPoint[vectorIndex_of_max_explained_var]) + ', at t=' + str(expVarIndRangeAfterSpike[vectorIndex_of_max_explained_var] * dt))
                    plt.subplot(5, 1, 4)
                    plt.plot(np.array([float(jj) for jj in expVarIndRangeAfterSpike[1:]]) * dt, explained_diffvar_at_each_set_of_endPoint)
                    plt.plot(float(expVarIndRangeAfterSpike[vectorIndex_of_max_diff_explained_var] + 1) * dt, explained_diffvar_at_each_set_of_endPoint[vectorIndex_of_max_diff_explained_var], 'xk', ms=16, lw=10)
                    plt.ylabel('d explained variance/dt')
                    plt.title('max devivative explained variance=' + str(explained_diffvar_at_each_set_of_endPoint[vectorIndex_of_max_diff_explained_var]) + ', at t=' + str(expVarIndRangeAfterSpike[vectorIndex_of_max_diff_explained_var] * dt))
                    plt.subplot(5, 1, 5)
                    plt.plot(np.array([float(jj) for jj in expVarIndRangeAfterSpike[2:]]) * dt, explained_diff2var_at_each_set_of_endPoint)
                    plt.plot(float(expVarIndRangeAfterSpike[vectorIndex_of_max_diff2_explained_var] + 2) * dt, explained_diff2var_at_each_set_of_endPoint[vectorIndex_of_max_diff2_explained_var], 'xk', ms=16, lw=10)
                    plt.ylabel('d explained variance/dt')
                    plt.title('max sencond derivative explained variance=' + str(explained_diff2var_at_each_set_of_endPoint[vectorIndex_of_max_diff2_explained_var]) + ', at t=' + str(expVarIndRangeAfterSpike[vectorIndex_of_max_diff2_explained_var] * dt))
            
                    #---plot of just explained variance
                    plt.figure()
                    plt.plot(np.array([float(jj) for jj in expVarIndRangeAfterSpike]) * dt, explained_var_at_each_set_of_endPoint)
                    plt.plot(float(expVarIndRangeAfterSpike[vectorIndex_of_max_explained_var]) * dt, explained_var_at_each_set_of_endPoint[vectorIndex_of_max_explained_var], 'xk', ms=16, lw=10)
                    plt.ylabel('Explained Variance', fontsize=20)
                    plt.xlabel('time after spike threshold (s)', fontsize=20)
            #        plt.title('Max explained variance='+ str(explained_var_at_each_set_of_endPoint[vectorIndex_of_max_explained_var])+', at t='+str(expVarIndRangeAfterSpike[vectorIndex_of_max_explained_var]*dt), fontsize=22)
#                    plt.title(cellTitle, fontsize=22)

                    plt.show(block=False)

                # --after finding the time point that minimizes the explained variance re-extract the voltage 
                # --at that time and make scatter plot of before and after voltages, and redo the linear regression and plot it
                # --I could be smart enough to index it correctly but I think it is safer to just redo it.
                all_v_spike_init_list = [v[all_thresholdInd[ii]] for ii, v in enumerate(all_v_spike_shape_list)]
            # USE THIS WHEN MUTIPLE VECTORS all_v_at_min_expVar_list=[v[list_of_endPointArrays[vectorIndex_of_max_explained_var][ii]] for ii, v in enumerate(all_v_spike_shape_list)] 
                all_v_at_min_expVar_list = [v[list_of_endPointArrays[vectorIndex_of_max_explained_var][ii]] for ii, v in enumerate(all_v_spike_shape_list)] 
            #    all_v_at_min_expVar_list=[v[vectorIndex_of_max_explained_var[ii]] for ii, v in enumerate(all_v_spike_shape_list)] 
            
            #    indNotExcluded_In_regress=list(np.setdiff1d(np.array([theInd for theInd in spikeIndDict['nonblip']]), np.array(waveIndOfFirstSpikes)))
            #    something is wrong with all_v_at_min_expVar_list--look at the difference between starting at .003 and .005 after thresh
                if DEBUG_MODE:
                    plt.figure(figsize=(20, 10))
                    plt.plot(all_v_spike_init_list, all_v_at_min_expVar_list, 'b.', ms=16, label='noise')  # list of voltage traces for blip
                    plt.xlabel('voltage at spike initiation (V)', fontsize=20)
                    plt.ylabel('voltage after spike (V)', fontsize=20)
#                    plt.title(cellTitle, fontsize=20)
            
                slope_at_min_expVar_list, intercept_at_min_expVar_list, r_value_at_min_expVar_list, p_value_at_min_expVar_list, std_err_at_min_expVar_list = \
                    stats.linregress(np.array(all_v_spike_init_list), np.array(all_v_at_min_expVar_list))

                print 'mean of voltage before spike', np.mean(all_v_spike_init_list)
                print 'mean of voltage after spike', np.mean(all_v_at_min_expVar_list)
                
                if DEBUG_MODE:
                    xlim = np.array([min(all_v_spike_init_list), max(all_v_spike_init_list)])
                    plotting.plotLineRegress1(slope_at_min_expVar_list, intercept_at_min_expVar_list, r_value_at_min_expVar_list, xlim)
                    plt.legend(loc=2, fontsize=20)

                def line_force_slope_to_1(x,c):
                    return x+c
                
                def line_force_int_to_0(x, m):
                    return m*x
                
                if slope_at_min_expVar_list>1.0:
                    warnings.warn('linear regression slope is bigger than one: forcing slope to 1 and refitting intercept.')
                    slope_at_min_expVar_list=1.0
                    (intercept_at_min_expVar_list, nothing)=curve_fit(line_force_slope_to_1, np.array(all_v_spike_init_list), np.array(all_v_at_min_expVar_list))
                    print "NEW INTERCEPT:", intercept_at_min_expVar_list
                    if intercept_at_min_expVar_list>0.0:
                        warnings.warn('/t ... and intercept is bigger than zero: forcing intercept to 0')
                        intercept_at_min_expVar_list=0.0
                    
                if intercept_at_min_expVar_list>0.0:
                    warnings.warn('Intercept is bigger than zero: forcing intercept to 0 and refitting slope.')
                    intercept_at_min_expVar_list=0.0                        
                    (slope_at_min_expVar_list, nothing)=curve_fit(line_force_int_to_0, np.array(all_v_spike_init_list), np.array(all_v_at_min_expVar_list))  
                    print "NEW SLOPE: ", slope_at_min_expVar_list  
                    if slope_at_min_expVar_list>1.0:
                        warnings.warn('/t ... and linear regression slope is bigger than one: forcing slope to 1.')
                        slope_at_min_expVar_list=1.0
                    

                if DEBUG_MODE:
                    xlim = np.array([min(all_v_spike_init_list), max(all_v_spike_init_list)])
                    plotting.plotLineRegressRed(slope_at_min_expVar_list, intercept_at_min_expVar_list, r_value_at_min_expVar_list, xlim)
                    plt.legend(loc=2, fontsize=20)
                    plt.show(block=False)                   
            
                #TODO:  Corinne look to see if these were calculated with zeroed out El if not does is matter?
            
                method_config['params'] = {
                        'a': slope_at_min_expVar_list,
                        'b': intercept_at_min_expVar_list
                    }
                if compute_spike_cut_length:
                    self.neuron_config['spike_cut_length'] = (list_of_endPointArrays[vectorIndex_of_max_explained_var][0])-int(sec_before_thresh_to_look_at_spikes/dt) #note this is dangerous if they arent' all at the same ind
            

            elif method_config['name'] == 'IandVbefore':
                method_config['params'] = {
                    'a': 1,
                    'b': 2,
                    'c': 3
                }
                raise Exception('IandVbefore of voltage_reset_method is not yet implemented')
            else:
                method_config['params'] = {}


        # configure the threshold_reset_method

        method_config = self.neuron_config['threshold_reset_method']        
        if method_config.get('params', None) is None:
            #TODO: rename from_paper to max or something
            if method_config['name'] == 'from_paper':
                method_config['params'] = { 'delta': const_to_add_to_thresh_for_reset }
            elif method_config['name'] == 'fixed':
                #TODO:this could probably also be renamed to something more descriptive like fixed_at_th_adapted
                method_config['params'] = { 'value': self.neuron_config['th_adapt'] - deltaV }
            elif method_config['name'] == 'V_plus_const':
                #TODO at some point this value my be fit
                method_config['params'] = {'value': const_to_add_to_thresh_for_reset}
            elif method_config['name'] == 'LIF':
                method_config['params'] = {'value': self.neuron_config['th_inf']}
            else:  
                method_config['params'] = {}


        #---------------------------------------------------------------------------
        #--------------calculate dynamics rules------------------------------------
        #--------------------------------------------------------------------------- 

        # configure the voltage dynamics method

        method_config = self.neuron_config['voltage_dynamics_method']
        if method_config.get('params', None) is None:
            print 'Configuring voltage_dynamics_methods in preprocessor.'
            if method_config['name'] == 'quadraticIofV':
                # --calculate non-linearities
                method_config['params'] = {'a': 1.10111713e-12,
                                           'b':4.33940334e-09,
                                           'c':-1.33483359e-07,
                                           'd':.015,
                                           'e':3.8e-11
                                           }
                raise Exception('quadraticIofV of voltage_dynamics_method preprocessing is not yet implemented')
            else:
                method_config['params'] = {}

        # configure the threshold dynamics method

        method_config = self.neuron_config['threshold_dynamics_method']
        if method_config.get('params', None) is None:
            print 'Configuring threshold_dynamics_methods in preprocessor.'
            if method_config['name'] == 'fixed':
                method_config['params'] = {
                    'value': self.neuron_config['th_adapt']
                }
                # TODO: perhaps this should be renames to adapted threshold since it is fixed at a particular place
            elif method_config['name'] == 'LIF':
                method_config['params'] = {'value': self.neuron_config['th_inf']}
            elif method_config['name'] == 'adapt_standard':
                method_config['params'] = {
                    'a': a_value,
                    'b': b_value 
                }
            else:
                method_config['params'] = {}

        # configure the ascurrent dynamics method
        method_config = self.neuron_config['AScurrent_dynamics_method']
        if method_config.get('params', None) is None:
            print 'Configuring AScurrent_dynamics_method in preprocessor.'
            if method_config['name'] == 'vector':
                method_config['params'] = {
                    'vector': [1, 2, 3]
                }
                # TODO: need to put in method here
                warnings.warn('this method is going to need care to make sure it is implemented correctly')
                raise Exception('vector of AScurrent_dynamics_method is not yet implemented')  
            elif method_config['name'] == 'expViaBlip':
                method_config['params'] = {}

                #must set asc_vector, acoeff_vector, tou, r, upper and lower bounds and
                #--calculate necessary voltages
                spike_voltage=self.superthreshold_blip_data['voltage'][0]
                spike_current=self.superthreshold_blip_data['current'][0] #used just for plotting
                noSpike_voltage=self.subthreshold_blip_data['voltage'][0]
                subtractVoltage=spike_voltage-noSpike_voltage

                #--get indicies for cutting                               
                spike_ind=find_spikes([spike_voltage], 'threshold', self.neuron_config['dt'])
                cut_ind=int(spike_ind[0])+self.neuron_config['spike_cut_length']  #NOTE: this is dependent on earlier calculations in reset rules
                
                cut_spike=spike_voltage[cut_ind:]
                ASvoltage=subtractVoltage[cut_ind:]
                
                #--Convert after spike voltage to after spike current 
                AScurrent=ASvoltage/self.neuron_config['R_input']

                time=np.arange(0, len(spike_voltage))*self.neuron_config['dt'] #used just for plotting 
                #cut_time=np.arange(0, len(cut_spike))*self.neuron_config['dt']
                
                #--Pad the end
                AScurrent_end_current=np.mean(AScurrent[-int(.02/self.neuron_config['dt']):])
                
                sec_to_pad=2.
                AScurrent=np.append(AScurrent, np.ones(int(sec_to_pad/self.neuron_config['dt']))*AScurrent_end_current)
                cut_time=np.arange(0, len(AScurrent))*self.neuron_config['dt']
                
                if DEBUG_MODE:
                    plt.figure()
                    plt.subplot(3,1,1)
                    plt.plot(time, spike_current)
                    plt.ylabel('current (pA)')
                    plt.title('Plots for AScurrent fitting', fontsize=20)
                
                    plt.subplot(3,1,2)
                    plt.plot(time, spike_voltage, 'b', label='spike')
                    plt.plot(time, noSpike_voltage, 'g', label='no spike')
                    plt.plot(time[cut_ind:], cut_spike, '--r', lw=2, label='portion used in fitting')
                    plt.ylabel('voltage (V)')
                
                    plt.subplot(3,1,3)
                    #subtract out leak
                    subtractVoltage=spike_voltage-noSpike_voltage
                    plt.plot(time, subtractVoltage, label='superthresh-subthresh V')
                    plt.plot(time[cut_ind:], subtractVoltage[cut_ind:], '--r', lw=2, label='portion used in fitting')
                    plt.xlabel('time (s)')
                    plt.ylabel('voltage (V)')
                    plt.legend()
                    plt.show(block=False)
                    
                    plt.figure()
                    plt.plot(cut_time, AScurrent)
                    plt.title('AS currents with pad')
                    plt.show(block=False)
                
                def exp_curve_4((t,const), a1, a2, a3, a4, k1, k2, k3, k4):
                    return a1*(np.exp(k1*t))+a2*(np.exp(k2*t))+a3*(np.exp(k3*t))+a4*(np.exp(k4*t))+const
                
                def exp_curve_3((t, const), a1, a2, a3, k1, k2, k3):
                    return a1*(np.exp(k1*t))+a2*(np.exp(k2*t))+a3*(np.exp(k3*t))+const
                
                def exp_curve_2((t, const), a1, a2, k1, k2):
                    return a1*(np.exp(k1*t))+a2*(np.exp(k2*t))+const
                
                p0_2=[5.e-12,-3e-12, -50, 1]
                p0_3=[1.3e-11, -3.99e-12, -8.69e-12,  -6.7e+01, -1.031,  -1.3e+01]
                p0_4=[1.23e-11, -6.17e-13, -7.22e-11, 6.53e-11, -7.01e+01,  -2.0, -9.090,  -8.46]

                
                # guessI_2=exp_curve_2(cut_time, p0_2[0], p0_2[1], p0_2[2], p0_2[3], p0_2[4])    
                # guessI_3=exp_curve_3(cut_time, p0_3[0], p0_3[1], p0_3[2], p0_3[3], p0_3[4], p0_3[5], p0_3[6])    
                # guessI_4=exp_curve_4(cut_time, p0_4[0], p0_4[1], p0_4[2], p0_4[3], p0_4[4], p0_4[5], p0_4[6], p0_4[7], p0_4[8]) 
                
                print "fitting 2 params"
                (popt_2, pcov_2)= curve_fit(exp_curve_2, (cut_time, AScurrent_end_current), AScurrent, p0=p0_2, maxfev=1000000)
                fitV_2=exp_curve_2((cut_time, AScurrent_end_current), popt_2[0], popt_2[1], popt_2[2], popt_2[3])
                print '2 parameters', popt_2
                
                print "fitting 3 params"
                (popt_3, pcov_3)= curve_fit(exp_curve_3, (cut_time, AScurrent_end_current), AScurrent, p0=p0_3, maxfev=1000000)
                fitV_3=exp_curve_3((cut_time, AScurrent_end_current), popt_3[0], popt_3[1], popt_3[2], popt_3[3], popt_3[4], popt_3[5])
                print '3 parameters', popt_3

                print "fitting 4 params"
                (popt_4, pcov_4)= curve_fit(exp_curve_4, (cut_time, AScurrent_end_current), AScurrent, p0=p0_4, maxfev=1000000)
                fitV_4=exp_curve_4((cut_time, AScurrent_end_current), popt_4[0], popt_4[1], popt_4[2], popt_4[3], popt_4[4], popt_4[5], popt_4[6], popt_4[7])    
                print '4 parameters', popt_4
                
                RSS_2=np.sum((AScurrent-fitV_2)**2)
                RSS_3=np.sum((AScurrent-fitV_3)**2)
                RSS_4=np.sum((AScurrent-fitV_4)**2)
                AIC_2=AIC.AICc(RSS_2, len(popt_2), len(AScurrent))
                AIC_3=AIC.AICc(RSS_3, len(popt_3), len(AScurrent))
                AIC_4=AIC.AICc(RSS_4, len(popt_4), len(AScurrent))
                
                AICvec=np.array([AIC_2, AIC_3, AIC_4])
                min_AIC_index=np.where(AICvec==min(AICvec))[0]
                
                if DEBUG_MODE:

                    
                    #---plotting just the AS voltage
#                    plt.figure()
#                    plt.subplot(2,1,1)
#                    plt.plot(cut_time, ASvoltage)
#                    plt.xlabel('time (s)')
#                    plt.ylabel('voltage (V)')
#                    plt.title('voltage due to AScurrent')
#                    
#                    #--plotting just the AS currents 
#                    plt.subplot(2,1,2)
#                    plt.plot(cut_time, AScurrent)
#                    plt.xlabel('time (s)')
#                    plt.ylabel('current (A)')
#                    plt.title('AScurrent via conversion from voltage')
#                    plt.show(block=False)
                    
                    #--plot optimization results
                    plt.figure()
                    plt.plot(cut_time, AScurrent, lw=2, label='data')
                    #plt.plot(cut_time, guessI_2,lw=1, label='2 exp guess')
                    plt.plot(cut_time, fitV_2, lw=2, label="2 exp fit: AIC=%.4g" % (AIC_2))
                    ##plt.plot(cut_time, guessI_3,lw=1, label='3 exp guess')
                    plt.plot(cut_time, fitV_3, '--', lw=2, label="3 exp fit: AIC=%.4g" % (AIC_3))
                    ##plt.plot(cut_time, guessI_4,lw=1, label='4 exp guess')
                    plt.plot(cut_time, fitV_4, lw=2, label="4 exp fit: AIC=%.4g" % (AIC_4))
                    plt.legend()
                    plt.show(block=False)                
                
                
                #TODO: Corinne--might want to penalize more exponentially more critically given how much time they add to fitting.
                if min_AIC_index==0:
                    self.neuron_config['asc_vector'] = popt_2[0:2]
                    self.neuron_config['tau'] = 1. / popt_2[2:]       
                elif min_AIC_index==1:
                    self.neuron_config['asc_vector'] = popt_3[0:3]
                    self.neuron_config['tau'] = 1. / popt_3[3:]
                elif min_AIC_index==2:
                    self.neuron_config['asc_vector'] = popt_4[0:4]
                    self.neuron_config['tau'] = 1. / popt_4[4:]
                else:
                    raise Exception('number of currents does not exist')
            
            elif method_config['name'] == 'expViaGLM':
                method_config['params'] = {}
                self.neuron_config['asc_vector'] = [3.e-12, 3.e-12]
                self.neuron_config['tau'] = [.01, .04]
            else:
                method_config['params'] = {}

        # configure the AScurrent_reset_method
        # this is down here because it depends on numbers computed for the AScurrent_dynamics_method
        method_config = self.neuron_config['AScurrent_reset_method']
        if method_config.get('params', {}) is None:
            if method_config['name'] == 'sum':
                method_config['params'] = {
                    'r': np.ones(len(self.neuron_config['tau']))
                }
            else:
                method_config['params'] = {}

        #----------------------------------------------------------------------
        #-----------following are optional methods that alter the data  -------
        #----------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#------------------for now I am commenting this out because I think this is now implemented------
#------------------automatically comming from MIES--maybe there should be a check to make--------
#------------------sure it is happening.  To do this we would need that list of indices --------
#------------------to make sure the current is zero there----------------------------------------
#        if self.optional_methods.get('cut_extra_current', False):
#            assert id_of_silence_list is not None, Exception("cut_extra_current requires an id_of_silence_list")
#
#            extra_current_ids = [ id_of_silence_list[ii] for ii in sweeps ]
#            cut_extra_current(self.stim_list, self.bio_list, extra_current_ids)  
   

        #----------------------------------------------------------------------
        #------------------finding spikes--------------------------------------
        #----------------------------------------------------------------------
        
        self.spike_ind_list = find_spikes(self.optimize_data['voltage'], spike_determination_method, dt)
        # plotting.plotSpikes(self.bio_list, self.spike_ind_list, self.neuron.dt, blockME=False, method=spike_determination_method)

        #---convert indices of spikes to times
        self.interpolated_spike_time_target_list = np.array(self.spike_ind_list) * dt

        # grid and interpolated target spike times are the same in the experimental data because there is only grid precision in the data
        self.grid_spike_time_target_list = self.interpolated_spike_time_target_list 
            
        #---find if any of the sweep arrays don't spike
        self.target_spike_mask = [ len(ind_list) > 0 for ind_list in self.spike_ind_list ]


        #------------
        # validation
        #------------

        spike_cut_length = self.neuron_config.get('spike_cut_length', None)
        assert spike_cut_length is not None, Exception("Spike cut length must be set, but it is not.")
        assert spike_cut_length >= 0, Exception("Spike cut length must be non-negative.")

    def preprocess_holdout_stimulus(self, sweeps, blip_sweeps, spike_determination_method):
        pass

def subsample_data(data, method, present_time_step, desired_time_step):
    if present_time_step > desired_time_step:
        raise Exception('you desired times step is smaller than your current time step')

    # number of elements to average over
    n = int(desired_time_step / present_time_step)

    data_subsampled = None

    if method == 'mean':
        # if n does not divide evenly into the length of the array, crop off the end
        end = n * int(len(data) / n)
            
        return np.mean(data[:end].reshape(-1,n), 1)

    raise Exception('unknown subsample method: %s' % (method))


#------------------for now I am commenting this out because I think this is now implemented------
#------------------automatically comming from MIES--------------------------------------------
# def cut_extra_current(stim_list, ind_list):
#    '''baseline current injection needed to keep cell at rest should be removed
#    a stimulus list is provided and one should be cut out
#    input:
#        ind_list: list of beginning and end index pairs per stim_sweep for which the rest should be calculated
#    '''
#
#    for ii, stim in enumerate(stim_list):
#        i_inject_at_rest=np.mean(stim[ind_list[ii][0]:ind_list[ii][1]])
#        stim_list[ii]=stim-i_inject_at_rest       

def zero_out_el_via_inj_current(stim, El_reference, El, input_resistance):
    '''Calculate how much current should be injected (using V=IR) to account for the variation in El 
    thoughout the experiment.  The El calculated in the blip current will be the reference El since
    this is where values for the intantaneous threshold will be.
    Input: 
        stim_list: list of current injection arrays to be altered
        El_reference: float that specifies the reference resting potential (usually the resting potential of the
                      the lowest amplitude spike evoking blip sweep
        El_list: List of foats that correspond to the resting potentials of the sweeps in the stim_list
        input_resistance: float the specifies the input resistence of the cells
    Returns: 
        Alters the stim_list.
    '''

    delta_v = El_reference - El
    delta_i = delta_v / input_resistance
    stim += delta_i

def zero_out_el_via_inj_current_oldWindOfSilence(stim_list, voltage_list, index_list, ref_volt, ref_ind, input_resistance):
    '''Calculate how much current should be injected (using V=IR) to account for the variation in El 
    thoughout the experiment.  The El calculated in the blip current will be the reference El since
    this is where values for the intantaneous threshold will be. 
    NOTE: MAKE SURE THIS IS BEING RUN AFTER ANY OTHER DATA PREPROCESSING STEPS
    '''
    # calculate El of blip current
    El_reference = np.mean(ref_volt[ref_ind])
    # calculate how much current should be injected using V=IR
    for jj, v_and_ind in enumerate(zip(voltage_list, index_list)):
        v, ind = v_and_ind
        El = np.mean(v[ind])
        delta_v = El_reference - El
        delta_i = delta_v / input_resistance
        stim_list[jj] = stim_list[jj] + delta_i

def align_and_cut_spikes(voltage_list, current_list, dt, sec_look_before_spike=0.002, sec_look_after_spike=.015, method='threshold'):
    ''' This function aligns the spikes to some criteria and returns a current and voltage trace of 
    of the spike over a time window.  Also returns the indices of the peaks, zero crossing,and threshold 
    in reference to the aligned spikes.
    
    Note: this function initially found peaks as a basis for the rest of the function.  However the peak
    algorithm is faulty with less than perfect data.  Therefore, it has been depreciated to only have a zero 
    crossing criterian 
    '''
    #TODO: CorinneT: fix all this hard coding and make sure it is zen with the calculations in the preprocessor
    ind_rangeBeforeCross_for_definingDerStd = np.array([.003, .0007]) / dt #TODO: wait does this work with only looking 2 ms default before spike above
    ind_rangeBeforeCross_for_threshFind = np.array([.0007 / dt, -.001 / dt])  # specify time before crossing to look for threshold crossing
    spike_shapes = []
    current_shapes = []
    index_before_spike = int(sec_look_before_spike / dt)
    index_after_spike = int(sec_look_after_spike / dt)
    postAligned_zeroCrossInd = np.array([])
    postAligned_thresholdInd = np.array([])
    spikeCount = 0
    spikeFromWhichSweep = []
    blip_v_arrayIndList = []
    nonblip_v_arrayIndList = []
    square_v_arrayIndList = []
    ramp_v_arrayIndList = []
    white_v_arrayIndList = []
    pink_v_arrayIndList = []
    brown_v_arrayIndList = []
    numOfSpikesInTrace = np.array([])
    zeroCrossInd_list = find_zero_cross(voltage_list)
    thresholdInd_list = get_threshold_ind(voltage_list, zeroCrossInd_list, ind_rangeBeforeCross_for_definingDerStd, ind_rangeBeforeCross_for_threshFind)  # this is kind of wierd at the moment because ind_rangeBeforeCross_for_threshFind is global
    for jj, voltage_AND_current_AND_zeroCross_AND_thresh in enumerate(zip(voltage_list, current_list, zeroCrossInd_list, thresholdInd_list)):
        voltage, current, wholeTrace_zeroCrossInd, wholeTrace_threshInd = voltage_AND_current_AND_zeroCross_AND_thresh
 
        if len(wholeTrace_threshInd) != len(wholeTrace_zeroCrossInd):
            raise Exception('the number of crosses and threshold indices found are not the same')
        
        numOfSpikesInTrace = np.append(numOfSpikesInTrace, len(wholeTrace_zeroCrossInd))
        
        if method is 'zeroCross':  # go backward from peak and find indicie where zero happens
            alignment_ind = wholeTrace_zeroCrossInd
            postAligned_zeroCrossInd = np.append(postAligned_zeroCrossInd, np.ones(len(wholeTrace_zeroCrossInd)) * index_before_spike)
            postAligned_thresholdInd = np.append(postAligned_thresholdInd, index_before_spike - (wholeTrace_zeroCrossInd - wholeTrace_threshInd)) 
        elif method is 'threshold':
            alignment_ind = wholeTrace_threshInd                        
            postAligned_zeroCrossInd = np.append(postAligned_zeroCrossInd, index_before_spike + wholeTrace_zeroCrossInd - wholeTrace_threshInd)
            postAligned_thresholdInd = np.append(postAligned_thresholdInd, np.ones(len(wholeTrace_zeroCrossInd)) * index_before_spike)
        else:
            raise Exception('Your alignment criteria is not defined')

        if len(alignment_ind) != len(wholeTrace_zeroCrossInd):
            raise Exception('the number of zeroCrossings and aligning criteria you are using are not the same')

        # print 'alignment_ind', alignment_ind
        spike_delimiters = [(ind - index_before_spike, ind + index_after_spike) for ind in alignment_ind]
        for d in spike_delimiters: 
            # this 'if' statement makes sure we don't cause a ValueError
            if min(d) > 0 and max(d) < len(voltage) - 1:
                spike_trace = voltage[d[0]:d[1]]
                current_trace = current[d[0]:d[1]]
                spike_shapes.append(spike_trace)
                current_shapes.append(current_trace)
#            #---add vectors-------
#                if jj in [0,6,7,17,27]:
#                    square_v_arrayIndList.append(spikeCount)
#                elif jj in [1, 2, 3]:
#                    blip_v_arrayIndList.append(spikeCount)
#                elif jj in [4,5]:
#                    ramp_v_arrayIndList.append(spikeCount)
#                elif jj in [8,9,10, 18,19,20, 28,29,30]:
#                    white_v_arrayIndList.append(spikeCount)
#                elif jj in [11,12,13, 21,22,23, 31,32,33]:
#                    pink_v_arrayIndList.append(spikeCount)
#                elif jj in [14,15,16, 24,25,26, 34,35,36]:
#                    brown_v_arrayIndList.append(spikeCount)
#                else:
#                    raise Exception('this file is not characterized')
#        
#                if jj not in [1, 2, 3]:
#                    nonblip_v_arrayIndList.append(spikeCount)
                    
                spikeCount = spikeCount + 1
                spikeFromWhichSweep.append(jj)

#    spikeIndDict={'blip': blip_v_arrayIndList, 'nonblip': nonblip_v_arrayIndList, 'square': square_v_arrayIndList, 
#                  'ramp': ramp_v_arrayIndList,'white': white_v_arrayIndList, 'pink': pink_v_arrayIndList, 
#                  'brown': brown_v_arrayIndList}
    
    # --error check for the the whole vectors
    if len(spike_shapes) != len(postAligned_zeroCrossInd) or len(postAligned_thresholdInd) != len(postAligned_zeroCrossInd):
        raise Exception('The number of values in peaks, zeroCross and threshold are not the same')

    if DEBUG_MODE:
        truncatedTime = np.arange(0, len(spike_shapes[0])) * dt
        plt.figure(figsize=(20, 10))
        for ii, spikewave in enumerate(spike_shapes):
            plt.plot(truncatedTime, spikewave)
            # plt.plot(truncatedTime[postAligned_peakInd[ii]],spikewave[postAligned_peakInd[ii]], '.k')
            plt.plot(truncatedTime[postAligned_zeroCrossInd[ii]], spikewave[postAligned_zeroCrossInd[ii]], 'xk')
            plt.plot(truncatedTime[postAligned_thresholdInd[ii]], spikewave[postAligned_thresholdInd[ii]], '*k')
            plt.xlabel('time (s)', fontsize=20)
            plt.ylabel('voltage (mV)', fontsize=20)
            
    # note: that depending on how things were aligned, all of one of the values will be the same.
    print "numOfSpikesInTrace", numOfSpikesInTrace
    temp = np.append(0, np.cumsum(numOfSpikesInTrace))  
    print 'temp', temp 
    waveIndOfFirstSpikes = [int(ii) for ii in list(temp[range(0, len(temp) - 1)])]         
    print "in cut spikes: waveIndOfFirstSpikes ", waveIndOfFirstSpikes
    return spike_shapes, current_shapes, postAligned_zeroCrossInd, postAligned_thresholdInd, waveIndOfFirstSpikes, spikeFromWhichSweep

def find_zero_cross(voltage_list):
    """
    Computes list of arrays of spike indices from list of voltage traces.
    The spike indices are computed using the upward zero crossing criterion. 
    
    :param voltage_list: list of arrays in units of volts.
    :returns: list of arrays of spike indices.
    
    .. note:: 
        In order to get the spike *times*, the returned spike index arrays 
        will need to be multiplied by dt.  This step will need to be done in 
        order to call functions from :mod:`fit_neuron.evaluate.spkd_lib`.
    """
    zeroCrossInd_list = []
    for voltage in voltage_list:
        v_array_len = len(voltage)
        # note: we do a reset the index AFTER the spikes 
        spike_ind = []            
        ind = -1 
        while ind < v_array_len - 1:          
            ind += 1              
            if voltage[ind] > 0: 
                # above_thresh_sample = voltage[ind:ind+25]
                # top_ind = above_thresh_sample.argmax() + ind
                spike_ind.append(ind)    
                # we keep pushing the index until the voltage comes back 
                # down, this ensures that we don't double count spikes 
                # ind = top_ind
                while voltage[ind] > -0.010 and ind < v_array_len - 1:  # changed -10 to -0.01 (SI units) 11-16-13
                    ind += 1                          
        spike_ind_arr = np.array(spike_ind)
        zeroCrossInd_list.append(spike_ind_arr)
        
    return zeroCrossInd_list

def get_threshold_ind(voltageTrace_list, zeroCross_list, ind_rangeForBaseLineCalc, ind_rangeForThreshSearch):
# def get_threshold_ind(voltageTrace_list, zeroCross_list, ind_rangeBeforeCross):  chenged from this to a situation were both the
# test interval is also specified for ease of plotting. note that other code fitting Jims data is not yet changed

    ''' Note that technically, you could use the peak instead of the zero crossing however then you 
    would probably want to make sure the time window you are looking behind is appropriate.  
    '''
    threshInd_list = []
#    kk=0
#    for voltage, zeroCross in zip(voltageTrace_list, zeroCross_list):
#
#        print kk
#        plt.figure()
#        plt.plot(voltage)
#        plt.title(str(kk))
#        plt.show()
#        kk+=1
    
    for voltage, zeroCross in zip(voltageTrace_list, zeroCross_list):
        
#        plt.figure()
#        plt.plot(voltage)
#        plt.plot(zeroCross, voltage[zeroCross])
#        plt.show()
        wholeTrace_threshInd = np.array([])
        for cross in zeroCross:
            dirCharInd = range(int(cross - ind_rangeForBaseLineCalc[0]), int(cross - ind_rangeForBaseLineCalc[1]) + 1)  # stop looking at derivative 1 ms before the cross
            derivative = np.diff(voltage[dirCharInd])
            max_preSpike_diff_v = np.max(derivative)
            std_preSpike_diff_v = np.std(derivative)
            testInd = range(int(cross - ind_rangeForThreshSearch[0]), int(cross - ind_rangeForThreshSearch[1] + 1))  # look for threshold between 1ms before cross to cross time
            testDer = np.diff(voltage[testInd])
#            plt.figure()
#            plt.plot(voltage)
#            plt.plot(dirCharInd, voltage[dirCharInd], 'r')
#            plt.plot(testInd, voltage[testInd], 'g')  
#            plt.title('in get_threshold_ind function in the preprocessor')
#            plt.show()          
            for ii in range(0, len(testDer) - 1):  # add -1 right here 2-20-2014
                thresholdFoundFlag = 0
                # 2-20-2014 jan 6 cell 2 was dying with and indicie error probably due to the not finding a threshold and making it to ii+1.  My guess is it will now error on the raise exception below
                if testDer[ii] > (max_preSpike_diff_v + std_preSpike_diff_v) and testDer[ii + 1] > testDer[ii]:
                    wholeTrace_threshInd = np.append(wholeTrace_threshInd, int(cross - ind_rangeForThreshSearch[0]) + ii)
                    thresholdFoundFlag = 1
                    break
                # if thresholdFoundFlag is 0:
                   # print 'at peak ind', peak, 'a voltage value larger than ', (max_preSpike_diff_v+std_preSpike_diff_v) ,' was not found between the peak ans 1 ms before the peak.'

        if len(wholeTrace_threshInd) != len(zeroCross):
            print 'number of threshold indices', len(wholeTrace_threshInd) 
            print 'number of zero crossing', len(zeroCross)
            raise Exception('the number of zeroCrossings and threshold indices found are not the same')
        threshInd_list.append(wholeTrace_threshInd)
        
    return threshInd_list  # note might also want to return actual threshold values

def find_spikes(voltage_list, method, dt):
    '''Calls the specified method to find the spikes.  
    input:
        voltage_list: list of numpy arrays.  Each experimental sweep is an array.
        method: string that specifies the method of defining a spike.
        dt: time step 
    output:
        list_of_spike_ind: a list of numpy arrays that specify the ind of the spikes. Each numpy array 
        will specify the indicies of the spikes for that sweep        
    '''
    
    #TODO: method might need to turn into another data type for the 
    #    threshod since I might have to specify a value along with it
    
    if method=='peaks':
        raise Exception('are you sure you want to be using peaks')
    elif method=='zeroCross':
        list_of_spike_ind=find_zero_cross(voltage_list)
    elif method=='threshold':
        zC_indList=find_zero_cross(voltage_list)

        ind_rangeBeforeCross_for_definingDerStd = np.array([.003, .0007]) / dt #TODO: wait does this work with only looking 2 ms default before spike above
        ind_rangeBeforeCross_for_threshFind = np.array([.0007 / dt, -.001 / dt])  # specify time before crossing to look for threshold crossing

        list_of_spike_ind=get_threshold_ind(voltage_list, zC_indList, ind_rangeBeforeCross_for_definingDerStd, ind_rangeBeforeCross_for_threshFind)
    else:
        raise Exception('the method for finding spikes is ill-defined')
    
    return list_of_spike_ind

def find_stimulus(current):
    start_idx = 0
    end_idx = 0
    
    length = len(current)
    for i in xrange(length):
        if current[i] != 0:
            start_idx = i
            break

    for i in xrange(length-1, -1, -1):
        if current[i] != 0:
            end_idx = i 
            break
    
    return start_idx, end_idx - start_idx + 1

def calculate_input_resistance_via_ramp(voltage, current, dt):
    '''When capacitance is zero then V=IR.  During a ramp the current and dV/dt are changing very slowly.'''
    stim_start, stim_len = find_stimulus(current)

    # .05% of the way into the ramp
    index = stim_start + int(stim_len/100)
    v_on_ramp = voltage[index]
    i_on_ramp = current[index]

    R_input = v_on_ramp / i_on_ramp

    if DEBUG_MODE:
        time=np.arange(0, len(current))*dt
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time,current)
        plt.plot(index*dt, current[index], 'xr', ms=20)
        plt.ylabel('current (A)', fontsize=16)
        plt.title('View index for resistance calculation', fontsize=20)
        plt.subplot(2, 1, 2)
        plt.plot(time, voltage)
        plt.ylabel('voltage (V)', fontsize=16)
        plt.plot(index*dt, voltage[index], 'xr', ms=20)
        plt.xlabel('time (s)', fontsize=16)
        plt.show(block=False)         

    return R_input

# --calculate capacitance from the subthreshold blip sweep
def calculate_capacitance_via_subthreshold_blip(voltage, current, dt):
    '''capacitance can be calculated by dumping a charge in a circuit.
    in this case C=Q/V.  Since Q=integral(I dt) which is equal to 
    I*delta_t'''

    stim_start, stim_len = find_stimulus(current)
    index = stim_start + int(stim_len/2)

    total_inj_current_during_blip = current[index] * .003
    max_voltage_index = stim_start + voltage[stim_start:].argmax()
    change_in_v_via_blip = voltage[max_voltage_index]

    cap = total_inj_current_during_blip / change_in_v_via_blip
    print 'capacitance', cap * 1e12, 'pF'

    if DEBUG_MODE:
        time=np.arange(0, len(current))*dt
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time, current)
        plt.plot(index*dt, current[index], 'xr', ms=20)
        plt.title('capacitance calc via subthreshold blip', fontsize=20)
        plt.ylabel('current (A)', fontsize=16)
        plt.subplot(2, 1, 2)
        plt.plot(time, voltage)
        plt.ylabel('voltage (V)', fontsize=16)
        plt.plot(max_voltage_index*dt, voltage[max_voltage_index], 'xr', ms=20)
        plt.xlabel('time (s)', fontsize=16)
        plt.show(block=False)

    return cap

def find_first_spike_voltage(voltage, dt):
    spike_ind_list_blip = find_spikes([voltage], 'threshold', dt)

    if DEBUG_MODE:
        plotting.plotSpikes([voltage], spike_ind_list_blip, dt, blockME=False, method='threshold')

    return voltage[spike_ind_list_blip[0][0]]

def calc_input_R_and_extrapolatedV_via_ramp(voltage, current, rampStartInd, dt, El):
        
    #--slope calculated from beginning of ramp and threshold 
    rampSpikeInd = find_spikes([voltage], 'threshold', dt)[0][0]
    v_at_spike=voltage[rampSpikeInd]
    I_at_spike=current[rampSpikeInd]  
    
    numOfPoints=rampSpikeInd-rampStartInd
        
    timeIncrements=.01
    measureInd=np.arange(rampStartInd, rampSpikeInd, int(timeIncrements/dt)).tolist()
    measureInd=[int(ii) for ii in measureInd]
    if measureInd[-1]!=rampSpikeInd:
        measureInd.append(rampSpikeInd)  #adding the index of the spike to the ind
    measured_dt=np.array(measureInd)*dt
    v_i=voltage[rampStartInd]
    i_i=current[rampStartInd]
    v_f=[np.mean(voltage[ii-5:ii+6]) for ii in measureInd]
    i_f=current[measureInd]

    #Find indexes of ramp between 20 and 40%    
    aveIndicies = range(int(.2*len(measureInd)), int(.4*len(measureInd))) 
#    chosen_R_ind=-10
    resistance=np.mean((np.array(v_f)[aveIndicies]-v_i)/(np.array(i_f)[aveIndicies]-i_i))
    
    #--HOWEVER WE CHOOSE R, WE ARE GOING TO DO A MAP AND CHANGE ALL VOLTAGE PARAMETERS
    #v2=R(I2-I1)+v1  NOTE V1 should be el
    v_extrapolated=resistance*I_at_spike+El
    print 'v_at_spike', v_at_spike
    deltaV=v_at_spike-v_extrapolated
    print 'difference in v', deltaV
    
    if DEBUG_MODE:
        rampVoltage=voltage[0:rampSpikeInd+10000]
        rampTime=np.arange(0, len(rampVoltage))*dt
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(rampTime, rampVoltage)
        plt.plot(rampStartInd*dt, v_i, '.r', ms=16)
        plt.plot(measured_dt, v_f, '.r', ms=16)        
        plt.ylabel('voltage (mV)')
        plt.title('New R calculation', fontsize=16)
        plt.subplot(3,1,2)
        plt.plot(measured_dt, ((v_f-v_i)/(i_f-i_i))*1e-6)
        plt.ylabel('Resistance (MOhms)')
        plt.subplot(3,1,3)
        plt.plot(rampTime, rampVoltage) #data
        plt.plot(rampTime[rampStartInd], v_i, '.b', ms=16) #start dot
#        plt.plot(rampTime[measureInd[chosen_R_ind]], rampVoltage[measureInd[chosen_R_ind]],  '.b', ms=16)
        plt.plot(rampTime[rampSpikeInd], v_extrapolated, 'xb', ms=12, lw=12, label='extrapolated') #extrapolated
        plt.plot([rampTime[rampStartInd], rampTime[rampSpikeInd]], [v_i, v_extrapolated], '--k', lw=4, label='R=%.3g' % (resistance))
        plt.plot(rampSpikeInd*dt, v_at_spike, '.r', ms=16, label='spike')
        plt.xlim([(rampStartInd-1000)*dt,(rampSpikeInd+2000)*dt])
        plt.legend()
        plt.ylabel('voltage (mV)')
        plt.xlabel('time (s)')
        plt.show(block=False)
        
    return resistance, deltaV

def calc_cap_via_fit_charge_dump(blip_voltage, blip_current, resistance, dt, El):
        
    #    time_blip=np.arange(0,len(blip_voltage))*dt
    #    plt.figure()
    #    plt.subplot(2,1,1)
    #    plt.plot(time_blip, blip_current)
    #    plt.ylabel('current (pA)')
    #    plt.subplot(2,1,2)
    #    plt.plot(time_blip, blip_voltage)
    #    plt.ylabel('voltage (mV)')
    #    plt.xlabel('time (s)')
    
    # calculate capacitance from subthreshold blip data
    stim_start, stim_len = find_stimulus(blip_current)
    index = stim_start + int(stim_len/2)
    
    total_inj_charge_during_blip = blip_current[index] * .003
    max_voltage_index = stim_start + blip_voltage[stim_start:].argmax()
    
    v2fit=blip_voltage[max_voltage_index:]-El
    t2fit=np.arange(0, len(v2fit))*dt
    
    def exp_curve_4(t, a1, a2, a3, a4, k1, k2, k3, k4):
        return a1*(np.exp(k1*t))+a2*(np.exp(k2*t))+a3*(np.exp(k3*t))+a4*(np.exp(k4*t))
    
    def exp_curve_3(t, a1, a2, a3, k1, k2, k3):
        return a1*(np.exp(k1*t))+a2*(np.exp(k2*t)+a3*np.exp(k3*t))
    
    def exp_curve_2(t, a1, a2, k1, k2):
        return a1*(np.exp(k1*t))+a2*(np.exp(k2*t))
    
    def exp_curve_1(t, a1, k1):
        return a1*(np.exp(k1*t))
    
    p0_1=[.08, -1000]
    p0_2=[.03, .01, -50, -5]
    p0_3=[.03, .02, .01, -100, -10, -5]
    p0_4=[.03, .02, .01, .01, -100, -10, -5, -1]
    
    guessV_1=exp_curve_1(t2fit, p0_1[0], p0_1[1])
    guessV_2=exp_curve_2(t2fit, p0_2[0], p0_2[1], p0_2[2], p0_2[3])    
    guessV_3=exp_curve_3(t2fit, p0_3[0], p0_3[1], p0_3[2], p0_3[3], p0_3[4], p0_3[5])    

    (popt_1, pcov_1)= curve_fit(exp_curve_1, t2fit, v2fit, p0=p0_1, maxfev=100000)    
    (popt_2, pcov_2)= curve_fit(exp_curve_2, t2fit, v2fit, p0=p0_2, maxfev=100000)
    (popt_3, pcov_3)= curve_fit(exp_curve_3, t2fit, v2fit, p0=p0_3, maxfev=100000)
    (popt_4, pcov_4)= curve_fit(exp_curve_4, t2fit, v2fit, p0=p0_4, maxfev=100000)

    print '1 parameters', popt_1
    print '2 parameters', popt_2
    print '3 parameters', popt_3
    print '4 parameters', popt_4

    fitV_1=exp_curve_1(t2fit, popt_1[0], popt_1[1])    
    fitV_2=exp_curve_2(t2fit, popt_2[0], popt_2[1], popt_2[2], popt_2[3])
    fitV_3=exp_curve_3(t2fit, popt_3[0], popt_3[1], popt_3[2], popt_3[3], popt_3[4], popt_3[5])    
    fitV_4=exp_curve_4(t2fit, popt_4[0], popt_4[1], popt_4[2], popt_4[3], popt_4[4], popt_4[5], popt_4[6], popt_4[7])    

    RSS_1=np.sum((v2fit-fitV_1)**2)    
    RSS_2=np.sum((v2fit-fitV_2)**2)
    RSS_3=np.sum((v2fit-fitV_3)**2)
    RSS_4=np.sum((v2fit-fitV_4)**2)
    
    AIC_1=AIC.AICc(RSS_1, len(popt_1), len(v2fit))
    AIC_2=AIC.AICc(RSS_2, len(popt_2), len(v2fit))
    AIC_3=AIC.AICc(RSS_3, len(popt_3), len(v2fit))
    AIC_4=AIC.AICc(RSS_4, len(popt_4), len(v2fit))

    AICvec=np.array([AIC_1, AIC_2, AIC_3, AIC_4])
    min_AIC_index=np.where(AICvec==min(AICvec))[0]
    print 'AIC', AICvec
    
    #--Now subtract out the fast component
    
    #HOW DO I DECIDE WHICH CURVE TO USE?
    
    #FOR THE MOMENT I AM CHOOSING TO ELIMINATE THE FAST EXP OF THE 4 FIT
    #find max k (which will give smallest tau)
    maxk=np.where(np.abs(popt_2)==np.max(np.abs(popt_2[2:])))[0][0]
    print 'maxk', maxk
    v_minus_fast=v2fit-exp_curve_1(t2fit, popt_2[maxk-2], popt_2[maxk])
    
    #now fit the cuve with one exp
    (popt_rmfast, pcov_rmfast)= curve_fit(exp_curve_1, t2fit, v_minus_fast, p0=p0_1)
    fitV_nonFast=exp_curve_1(t2fit, popt_rmfast[0], popt_rmfast[1])
    
    #cut off fast part of curve
    (popt_trunc, pcov_trunc)= curve_fit(exp_curve_1, t2fit[int(.007/dt):], v2fit[int(.007/dt):], p0=p0_1)
    fitV_trunc=exp_curve_1(t2fit, popt_trunc[0], popt_trunc[1])
    
    if DEBUG_MODE:

        plotInd=range(0,int(.04/dt))
        
        plt.figure()
        plt.plot(t2fit[plotInd], v2fit[plotInd], lw=8, label="data: C via charge dump=%.3g" % (total_inj_charge_during_blip/v2fit[0]))
    #    plt.plot(t2fit, guessV_1,lw=5, label='1 exp guess')
        plt.plot(t2fit[plotInd], fitV_1[plotInd], lw=6, 
                 label="1 exp: C via charge dump=%.3g, C via tau=%.3g" % (total_inj_charge_during_blip/fitV_1[0],-1./(popt_1[1]*resistance)))    
        plt.plot(t2fit[plotInd], fitV_2[plotInd], lw=4, 
                 label="2 exp: C via charge dump=%.3g" % (total_inj_charge_during_blip/fitV_2[0]))
        #plt.plot(t2fit, guessV_3,lw=1, label='3 exp guess')
    #    plt.plot(t2fit[plotInd], fitV_3[plotInd], lw=2, label="3 exp fit: AIC=%.4g" % (AIC_3))
    #    plt.plot(t2fit[plotInd], fitV_4[plotInd], '--', lw=2, label="4 exp fit: AIC=%.4g" % (AIC_4))
        plt.xlabel('time (s)')
        plt.ylabel('voltage (V)')
        plt.title('fitting voltage of subthreshold blip', fontsize=16)
        plt.plot(t2fit[plotInd], v_minus_fast[plotInd], '--', lw=2, 
                 label="data-fast exp: C via charge dump=%.3g" % (total_inj_charge_during_blip/v_minus_fast[0]))
        plt.plot(t2fit[plotInd], fitV_nonFast[plotInd], lw=2, 
                 label="single exp fit to: data-fast exp: C via charge dump=%.3g, C via tau=%.3g"  % (total_inj_charge_during_blip/fitV_nonFast[0], -1./(popt_rmfast[1]*resistance)))
        plt.plot(t2fit[plotInd], fitV_trunc[plotInd], lw=2, 
                 label="single exp fit to truncated (.0007 s) data: C via charge dump=%.3g, C via tau=%.3g" % (total_inj_charge_during_blip/fitV_trunc[0], -1./(popt_trunc[1]*resistance)))    
    #    plt.plot(t2fit[plotInd], v_slow[plotInd], lw=2, label="slow exp: C=%.3g" % (total_inj_charge_during_blip/v_slow[0]))
        plt.legend()
        plt.show(block=False)
    
    #for now I will use C from truncated 1 exponential fit via charge dump
    return total_inj_charge_during_blip/fitV_trunc[0], total_inj_charge_during_blip

def calculate_input_R_via_subthresh_noise(voltage, current, El, start_idx, dt):
    
    print 'EL FOR RAM:', El
    '''requires subthreshold noise'''
    if find_spikes([voltage], 'threshold', dt)[0]:
        raise Exception('there is a spike in your subthreshold noise')
    
    #find end of stimulus
    length=len(current)
    for i in xrange(length-1, -1, -1):
        if current[i] != 0:
            end_idx = i 
            break
    
    #find region of stimulus
    aveV=np.mean(voltage[start_idx: end_idx])-El
    aveI=np.mean(current[start_idx: end_idx])
    
    resistance=-aveV/aveI
    return resistance

def R_via_subthresh_linRgress(voltage, current, dt):
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
    print "R via least squares", resistance*1e-6, "Mohms"
    print "C via least squares", capacitance*1e12, "pF"
    print "El via least squares", El*1e3, "mV" 
    print "tau", resistance*capacitance*1e3, "ms"
    return resistance, capacitance, El

def RC_via_subthresh_GLM(voltage, current, dt):
    '''This will only work with subthreshold noise
    '''   
    #Square filter for smoothing
    ts = 1000.*1e-6
    sq_filt = np.ones((round(ts/dt,1)))
    sq_filt = np.squeeze(sq_filt/len(sq_filt))
    
    #Initial tau and C as seed inputs for optimization
    cinit = 50.*1e-12
    tauinit = 20.*1e-3
    
    start_idx = 41000; tend = 99000  # HARD-CODED SO LOOK OUT
    
    #Unsmoothed raw data
    El = voltage[0]      #resting potential
    v = voltage[start_idx:tend-1]
    vs = voltage[start_idx+1:tend]
    dvdt = (vs-v)/dt
    i = current[start_idx:tend-1]
    
    #Smooth data
    sm_vL = El
    sm_v = np.convolve(sq_filt,v,'same')
    sm_i = np.convolve(i,sq_filt,'same')
    sm_dv = np.convolve(dvdt,sq_filt,'same')
    
    #Input colmn vectors to GLM
    inp = np.zeros((tend-start_idx-1,2))
    inp[:,0] = -(sm_v-sm_vL)/tauinit
    inp[:,1] = i/cinit
    inp = sm.add_constant(inp, prepend = False)
    
    #Output data to which model is fit
    out = sm_dv
    
    #Fit GLM
    glm_fit = sm.GLM(out[10:-9],inp[10:-9,:],family=sm.families.Gaussian(sm.families.links.identity))
    
    #GLM fit results
    res = glm_fit.fit()    
    fitprs = res.params    
    fit_dvdt = res.mu         
    fit_v = np.cumsum(fit_dvdt*dt)
    
    #COmpute TAU and C
    tau = tauinit/fitprs[0]
    capacitance = cinit/fitprs[1]
    resistance=tau/capacitance
    
    #Compare derivative fits
#    plt.plot(out[10:-9],lw = 0.5)
#    plt.plot(fit_dvdt,lw = 2)
#    plt.show()
#    
#    #Compare actual voltages
#    plt.plot(-(inp[10:-9,0]-inp[10,0])*tauinit)
#    plt.plot(fit_v,lw = 2)
#    plt.show()
    return resistance, capacitance

def calc_a_b_from_muliblip(multi_SS, dt):

    def exp_force_c((t, const), a1, k1):
        return a1*(np.exp(k1*t))+const

    def exp_fit_c(t, a1, k1, const):
        return a1*(np.exp(k1*t))+const

    multi_SS_v=multi_SS['voltage']
    multi_SS_i=multi_SS['current']
    
    spike_ind=find_spikes(multi_SS_v, 'threshold', dt)
    
    #SHOULD I REMOVE SPIKES OR JUST ERROR THE NEURON IF THINGS ARE SPIKING OUT OF ORDER?
    
    
    #these are what I want to be final
    time_previous_spike=[]
    threshold=[]
    thresh_first_spike=[]  #will set constant to this
    
    if DEBUG_MODE:
        plt.figure()
    for k in range(0, len(multi_SS_v)):
        thresh=[multi_SS_v[k][j] for j in spike_ind[k]]
        if thresh!=[] and len(thresh)>1:# there needs to be more than one spike so that we can find the time difference
            thresh_first_spike.append(thresh[0])
            threshold.append(thresh[1:])
            time_before_temp=[]
            for j in range(1,len(thresh)):
                time_before_temp.append((spike_ind[k][j]-spike_ind[k][j-1])*dt)
        #for each number calculate and distance from the first spike   
            time_previous_spike.append(time_before_temp)    
        if DEBUG_MODE:
            plt.subplot(2,1,1)
            plt.plot(np.arange(0, len(multi_SS_i[k]))*dt, multi_SS_i[k]*1e12, lw=2)
            plt.ylabel('current (pA)', fontsize=16)
            plt.subplot(2,1,2)
            plt.plot(np.arange(0, len(multi_SS_v[k]))*dt, multi_SS_v[k], lw=2)
            plt.plot(spike_ind[k]*dt, thresh, '.k', ms=16)
            plt.ylabel('voltage (mV)', fontsize=16)
            plt.xlabel('time (s)', fontsize=16)

    
    #put numbers in one vector5
    thresh_inf=np.mean(thresh_first_spike)
    print "average threshold of first spike",  thresh_inf
    threshold=np.concatenate(threshold)
    time_previous_spike=np.concatenate(time_previous_spike)  #note that this will have nans in it
    
#    for thr, times in zip(threshold, time_previous_spike):
#        print thr
#        print times
#        if len(thr)!=len(times):
#            print "not equal", 

    if DEBUG_MODE:
        plt.figure()
        plt.plot(time_previous_spike, threshold, '.k', ms=16)
        plt.ylabel('threshold (mV)')
        plt.xlabel('time after last spike (s)')
    
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
    if DEBUG_MODE:
        plt.plot(time_previous_spike, fit_force, 'r', lw=4, label='exp fit (force const to thesh first spike)')
        plt.plot(time_previous_spike, fit_fit, 'b', lw=4, label='exp fit (fit constant)')
        plt.legend()
        
        plt.show(block=False)
    
    #TODO: Corinne abs is put in hastily make sure this is ok
    const_to_add_to_thresh_for_reset=abs(popt_force[0]) 
    b=abs(popt_force[1])
    return const_to_add_to_thresh_for_reset, b

def calc_deltaV_via_specified_R_and_ramp(resistance, voltage, current, rampStartInd, dt, El):

    #--slope calculated from beginning of ramp and threshold 
    rampSpikeInd = find_spikes([voltage], 'threshold', dt)[0][0]
    v_at_spike=voltage[rampSpikeInd]
    I_at_spike=current[rampSpikeInd]  
    v_i=voltage[rampStartInd]
    
    #--HOWEVER WE CHOOSE R, WE ARE GOING TO DO A MAP AND CHANGE ALL VOLTAGE PARAMETERS
    #v2=R(I2-I1)+v1  NOTE V1 should be el
    v_extrapolated=resistance*I_at_spike+El
    print 'v_at_spike', v_at_spike
    deltaV=v_at_spike-v_extrapolated
    print 'difference in v', deltaV
    
    if DEBUG_MODE:
        rampVoltage=voltage[0:rampSpikeInd+10000]
        rampTime=np.arange(0, len(rampVoltage))*dt
        plt.figure()
        plt.plot(rampTime, rampVoltage) #data
        plt.plot(rampTime[rampStartInd], v_i, '.b', ms=16) #start dot
#        plt.plot(rampTime[measureInd[chosen_R_ind]], rampVoltage[measureInd[chosen_R_ind]],  '.b', ms=16)
        plt.plot(rampTime[rampSpikeInd], v_extrapolated, 'xb', ms=12, lw=12, label='extrapolated') #extrapolated
        plt.plot([rampTime[rampStartInd], rampTime[rampSpikeInd]], [v_i, v_extrapolated], '--k', lw=4, label='R=%.3g' % (resistance))
        plt.plot(rampSpikeInd*dt, v_at_spike, '.r', ms=16, label='spike')
        plt.xlim([(rampStartInd-1000)*dt,(rampSpikeInd+2000)*dt])
        plt.title('calculation of delta V with specified Resistance', fontsize=20)
        plt.legend()
        plt.ylabel('voltage (mV)')
        plt.xlabel('time (s)')
        plt.show(block=False)
        
    return deltaV
