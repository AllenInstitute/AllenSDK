import logging

import numpy as np
import copy
import warnings
import json, utilities

from neuron_methods import GLIFNeuronMethod, METHOD_LIBRARY

class GLIFNeuronException( Exception ):
    def __init__(self, message, data):
        super(Exception, self).__init__(message)
        self.data = data
    
class GLIFNeuron( object ):    
    '''Generalized Linear Integrate and Fire neuron
    '''
    TYPE = "GLIF"

    def __init__(self, El, dt, tau, R_input, C, asc_vector, spike_cut_length, th_inf, coeffs,
                 AScurrent_dynamics_method, voltage_dynamics_method, threshold_dynamics_method,
                 AScurrent_reset_method, voltage_reset_method, threshold_reset_method,
                 init_voltage, init_threshold, init_AScurrents): 

        self.type = GLIFNeuron.TYPE
        self.El = El
        self.dt = dt
        self.tau = np.array(tau)
        self.R_input = R_input
        self.C = C
        self.asc_vector = np.array(asc_vector)
        self.spike_cut_length = int(spike_cut_length)
        self.th_inf = th_inf
        
        self.init_voltage = init_voltage
        self.init_threshold = init_threshold
        self.init_AScurrents = init_AScurrents

        assert len(tau) == len(asc_vector), Exception("After-spike current vector must have same length as tau (%d vs %d)" % (asc_vector, tau))
        assert len(self.init_AScurrents) == len(self.tau), Exception("init_AScurrents length (%d) must have same length as tau (%d)" % (len(self.init_AScurrents), len(self.tau)))


        # values computed based on inputs
        self.k = 1.0 / self.tau
        self.G = 1.0 / self.R_input

        # Values that can be fit: They scale the input values.  
        # These are allowed to have default values because they are going to get optimized.
        self.coeffs = {
            'th_inf': 1,
            'C': 1,
            'G': 1,
            'b': 1,
            'a': 1,
            'asc_vector': np.ones(len(self.tau))
        }

        self.coeffs.update(coeffs)
        
        logging.debug('spike cut length: %d' %  self.spike_cut_length)

        # initialize dynamics methods
        self.AScurrent_dynamics_method = self.configure_library_method('AScurrent_dynamics_method', AScurrent_dynamics_method)
        self.voltage_dynamics_method = self.configure_library_method('voltage_dynamics_method', voltage_dynamics_method)
        self.threshold_dynamics_method = self.configure_library_method('threshold_dynamics_method', threshold_dynamics_method)

        # initialize reset methods
        self.AScurrent_reset_method = self.configure_library_method('AScurrent_reset_method', AScurrent_reset_method)
        self.voltage_reset_method = self.configure_library_method('voltage_reset_method', voltage_reset_method)
        self.threshold_reset_method = self.configure_library_method('threshold_reset_method', threshold_reset_method)

    def __str__(self):
        return json.dumps(self.to_dict(), default=utilities.json_handler, indent=2)

    @staticmethod
    def from_dict(d):
        return GLIFNeuron(El = d['El'],
                          dt = d['dt'],
                          tau = d['tau'],
                          R_input = d['R_input'],
                          C = d['C'],
                          asc_vector = d['asc_vector'],
                          spike_cut_length = d['spike_cut_length'],
                          th_inf = d['th_inf'],
                          coeffs = d.get('coeffs', {}),
                          AScurrent_dynamics_method = d['AScurrent_dynamics_method'],
                          voltage_dynamics_method = d['voltage_dynamics_method'],
                          threshold_dynamics_method = d['threshold_dynamics_method'],
                          voltage_reset_method = d['voltage_reset_method'],
                          AScurrent_reset_method = d['AScurrent_reset_method'],
                          threshold_reset_method = d['threshold_reset_method'],
                          init_voltage = d['init_voltage'],
                          init_threshold = d['init_threshold'],
                          init_AScurrents = d['init_AScurrents'])

    def to_dict(self):
        return {
            'type': self.type,
            'El': self.El,
            'dt': self.dt,
            'tau': self.tau,
            'R_input': self.R_input,
            'C': self.C,
            'asc_vector': self.asc_vector,
            'spike_cut_length': self.spike_cut_length,
            'th_inf': self.th_inf,
            'coeffs': self.coeffs,
            'AScurrent_dynamics_method': self.AScurrent_dynamics_method,
            'voltage_dynamics_method': self.voltage_dynamics_method,
            'threshold_dynamics_method': self.threshold_dynamics_method,
            'AScurrent_reset_method': self.AScurrent_reset_method,
            'voltage_reset_method': self.voltage_reset_method,
            'threshold_reset_method': self.threshold_reset_method,
            'init_voltage': self.init_voltage,
            'init_threshold': self.init_threshold,
            'init_AScurrents': self.init_AScurrents
        }

    def configure_method(self, method_name, method, method_params):
        return GLIFNeuronMethod(method_name, method, method_params)

    def configure_library_method(self, method_type, params):
        method_options = METHOD_LIBRARY.get(method_type, None)

        assert method_options is not None, Exception("Unknown method type (%s)" % method_type)
        
        method_name = params.get('name', None)
        method_params = params.get('params', None)
        
        assert method_name is not None, Exception("Method configuration for %s has no 'name'" % (method_type))
        assert method_params is not None, Exception("Method configuration for %s has no 'params'" % (method_params))
        
        method = method_options.get(method_name, None)
        
        assert method is not None, Exception("unknown method name %s of type %s" % (method_name, method_type))
        
        return self.configure_method(method_name, method, method_params)

    def dynamics(self, voltage_t0, threshold_t0, AScurrents_t0, inj, time_step, spike_time_steps):    
        '''Impliments the current based Mihalas Neiber GLIF neuron calculating variable 
        values (voltage, threshold and after spike currents) at 1 time step.
        Inputs:
            voltage_t0: scalar value of voltage threshold 
            threshold_t0: scalar value of voltage threshold
            AScurrents_t0: vector of scalar values for the after spike currents
            inj: scalar value of external current injection at one time step
        Returns: the scalar voltage (voltage_t1), scalar voltage threshold 
           (threshold_t1), and a array of scalar after spike currents (AScurrents_t1)
           as the result of the initial values and a current injection.
        '''

        AScurrents_t1 = self.AScurrent_dynamics_method(self, AScurrents_t0, time_step, spike_time_steps)
        voltage_t1 = self.voltage_dynamics_method(self, voltage_t0, AScurrents_t0, inj)
        threshold_t1 = self.threshold_dynamics_method(self, threshold_t0, voltage_t0)

        return voltage_t1, threshold_t1, AScurrents_t1
    
        
    #-------------------------------------------------------------------
    #----------RESET RULES----------------------------------------------
    #-------------------------------------------------------------------
    def reset(self, voltage_t0, threshold_t0, AScurrents_t0):
        '''The purpose is to reset the variables of the Mihalas-Neiber neuron.  Note: In Stefan's original code their are 
        2 MNupdate functions with the only difference being the t variable (where in one it was t and the other it
        was neuron.dt) to used when the times were found in the grid versus extrapolation.  Here I just have one function
        where the dt has to be entered  
        Inputs:
           p:  contains a bunch of neuron values
           param: parameters [C, G, a, ie, ii] correspond to the parameters in Stefan's paper
           var_in: [v,th,ie,ii]: variables that are evolving in time
           dt: delta t of each time step
        Returns:
           reset values of scalar voltage (voltage_t1), scalar threshold (threshold_t1), and an array
             of the after spike currents (AScurrents_t1) after there is a spike'''
        
        AScurrents_t1 = self.AScurrent_reset_method(self, AScurrents_t0)  #TODO: David I think you want to feed in r here
        voltage_t1 = self.voltage_reset_method(self, voltage_t0)
        threshold_t1 = self.threshold_reset_method(self, threshold_t0, voltage_t1)

        if voltage_t1 > threshold_t1:
            Exception("Voltage reset above threshold at time step (%f): voltage_t1 (%f) threshold_t1 (%f), voltage_t0 (%f) threshold_t0 (%f) AScurrents_t0 (%s)" % (t, voltage_t1, threshold_t1, voltage_t0, threshold_t0, repr(AScurrents_t0)))

        return voltage_t1, threshold_t1, AScurrents_t1
    


    #-----------------------------------------------------------------------------------
    #------------run functions----------------------------------------------------------
    #-----------------------------------------------------------------------------------
    def run(self, stim):
        '''Steps though dynamics equations.  After each step check if threshold is larger than 
        voltage.  If it is the next values in the train are set to reset values.  NANS may or may 
        not be injected based on flag that specifies if spikes are cut.
        
        @param voltage_t0: scalar voltage that the model starts at
        @param threshold_t0: scalar threshold the modeinterpolatedSpikeVoltage_list, interpolated_spike_threshold_listl starts at  TODO: elaborate on this
        @param AScurrents_t0: vector of initial scalar values of after spike currents
        @param stim: vector of scalar current values
        @return: tuple of voltage_out scalar and grid_spike_time  TODO: elaborate on this
        '''

        voltage_t0 = self.init_voltage
        threshold_t0 = self.init_threshold
        AScurrents_t0 = self.init_AScurrents

        num_time_steps = len(stim) 
        num_AScurrents = len(AScurrents_t0)
        
        # pre-allocate the output voltages, thresholds, and after-spike currents
        voltage_out=np.empty(num_time_steps)
        voltage_out[:]=np.nan
        threshold_out=np.empty(num_time_steps)
        threshold_out[:]=np.nan
        AScurrents_out=np.empty(shape=(num_time_steps, num_AScurrents))
        AScurrents_out[:]=np.nan        

        # array that will hold spike indices
        spike_time_steps = []
        grid_spike_times = []
        interpolated_spike_times = []
        interpolated_spike_voltage = []
        interpolated_spike_threshold = []

        time_step = 0
        while time_step < num_time_steps:
            if time_step % 10000 == 0:
                logging.info("time step %d / %d" % (time_step,  num_time_steps))

            # compute voltage, threshold, and ascurrents at current time step
            (voltage_t1, threshold_t1, AScurrents_t1) = self.dynamics(voltage_t0, threshold_t0, AScurrents_t0, stim[time_step], time_step, spike_time_steps) 

            #if the voltage is bigger than the threshold record the spike and reset the values
            if voltage_t1 > threshold_t1: 
                '''
                THINK ABOUT DEFINITION HERE SHOULD THE SPIKE BE COUNTED AND RESET BE DEFINED AND HOW THAT WILL AFFECT THE 
                CALCULATION IN REFERENCE TO THE SPIKES. OPTIONS ARE:
                        1. THRESHOLD IS ABOVE, THAT PRESENT VALUE GETS SET TO RESET AND THE INDEX IS COUNTED AT PRESENT POINT
                            Rational: at some point before the present value the voltage crossed threshold.  So the present point 
                            should be reset, the spike index should be set to the present value. The interpolated time would be 
                            before the present spike.  In this case when running the run_until_spike
                            would you have to let it run one extra ind to see if it spikes? (this is the method I am using now).
                        2. THRESHOLD IS ABOVE, THE NEXT VALUE GETS SET TO RESET AND THE INDEX IS COUNTED AT THE PRESENT POINT
                            Rational: Here the the voltage above threshold gets recorded and the spikeInd is recorded at the 
                            present ind, and the reset is at the next ind.  In this case the interpolated time will be before 
                            the spike ind. Would you be getting an extra spike here? 
                        3. THRESHOLD IS ABOVE, THE NEXT VALUE GETS SET TO RESET AND THE INDEX IS COUNTED AT THE NEXT POINT.
                            Rational:  I don't this this makes sense.  since the spike really happened at least at the index
                            before the reset and actually sometime before that.
                '''
                # spike_time_steps are stimulus indices when voltage surpassed threshold
                spike_time_steps.append(time_step)
                grid_spike_times.append(time_step * self.dt) 

                # compute higher fidelity spike time/voltage/threshold by linearly interpolating 
                interpolated_spike_times.append(interpolate_spike_time(self.dt, time_step, threshold_t0, threshold_t1, voltage_t0, voltage_t1))

                interpolated_spike_time_offset = interpolated_spike_times[-1] - (time_step - 1) * self.dt
                interpolated_spike_voltage.append(interpolate_spike_value(self.dt, interpolated_spike_time_offset, voltage_t0, voltage_t1))
                interpolated_spike_threshold.append(interpolate_spike_value(self.dt, interpolated_spike_time_offset, threshold_t0, threshold_t1))
            
                # reset voltage, threshold, and after-spike currents
                (voltage_t0, threshold_t0, AScurrents_t0) = self.reset(voltage_t1, threshold_t1, AScurrents_t1) 

                # if we are not integrating during the spike (which includes right now), insert nans then jump ahead
                if self.spike_cut_length > 0:
                    n = self.spike_cut_length
                    voltage_out[time_step:time_step+n] = np.nan
                    threshold_out[time_step:time_step+n] = np.nan
                    AScurrents_out[time_step:time_step+n,:] = np.nan

                    time_step += self.spike_cut_length
                else:
                    # we are integrating during the spike, so store the reset values
                    voltage_out[time_step] = voltage_t0 
                    threshold_out[time_step] = threshold_t0
                    AScurrents_out[time_step,:] = AScurrents_t0

                    time_step += 1
            else:
                # there was no spike, store the next voltages
                voltage_out[time_step] = voltage_t1 
                threshold_out[time_step] = threshold_t1
                AScurrents_out[time_step,:] = AScurrents_t1

                voltage_t0 = voltage_t1
                threshold_t0 = threshold_t1
                AScurrents_t0 = AScurrents_t1
                
                time_step += 1

        return {
            'voltage': voltage_out, 
            'threshold': threshold_out, 
            'AScurrents': AScurrents_out,
            'grid_spike_times': np.array(grid_spike_times), 
            'interpolated_spike_times': np.array(interpolated_spike_times), 
            'spike_time_steps': np.array(spike_time_steps), 
            'interpolated_spike_voltage': np.array(interpolated_spike_voltage), 
            'interpolated_spike_threshold': np.array(interpolated_spike_threshold)
            }


    def run_with_biological_spikes(self, stim, bio_spike_time_steps):
        '''this functions takes an array of spike time indices and runs the model from where each of the spikes happen
        input:
            voltage_t0: scalar of the initial voltage
            threshold_t0: scalar of initial threshold
            AScurrents_t0: array of initial after spike currents
            stim:  array with the stimulus at each time
            spike_time_steps: array of indices of the spikes (in reference to the local stimulus) that will be the start points for the model
            interpolated_spike_times: array of interpolated spike times of the target trace
        Output
            actualTimeArray: an array of the actual times of the spikes. NOTE: THESE TIMES ARE CALCULATED BY ADDING THE TIME OF 
            THE INDIVIDUAL SPIKE TO THE TIME OF THE LAST SPIKE. 
            grid_spike_times: an array of the time of the spikes with the grid time precision.NOTE: THESE TIMES ARE CALCULATED BY A
            ADDING THE TIME OF THE INDIVIDUAL SPIKE TO THE TIME OF THE LAST SPIKE.
            voltage: array of voltage values. NOTE: IF THE MODEL NEURON SPIKES BEFORE THE TARGET THE VOLTAGE WILL 
                NOT BE CALCULATED THEREFORE THE RESULTING VECTOR WILL NOT BE AS LONG AS THE TARGET AND ALSO WILL NOT 
                MAKE SENSE WITH THE STIMULUS UNLESS YOU CUT IT AND OUTPUT IT TOO.
            gridISIFromLastTargSpike_array:  array of spike times of the model in reference to the last target (biological) spike 
                (not in reference to sweep start)
            grid_spike_voltage:  array of scalars that contain the voltage of the model neuron when the target or bio neuron spikes.    
            grid_spike_threshold: array of scalars that contain the threshold of the model neuron when the target or bio neuron spikes. 
            NOTE: grid_spike_voltage may be larger than the corresponding values in the grid_spike_threshold because the model
                still runs until the time of the target or bio spike even if the model has already spiked.
            SIGNIFICANT CAVEATS: There are two situatations where the storage of a spike and/or how it should be punished is unclear
                1.  The biological or target spike train does not fire throughout the entire course of the sweep but the model does.
                2.  In the region of stimulus after the last biological spike, the model spikes.
                  In both these situations there are three options.
                    a.  Throw out the data.  It seems like valuable information could be disregarded in this situation.  
                    b.  Insert a virtual reference spike somewhere.  This is nonideal because you don't know where the 
                        real neuron would have spiked and so the weight of you punishment will be incorrect.  However,
                        if this is done at the end of the sweep would be a reasonable choice because if the real neuron were
                        going to produce a spike, it is clearly past the end of the current injection.  However, you don't want
                        to set the virtual spike out to infinity because then the punishment would be disproportionately large 
                        compared to the punishment of other spikes.  The end of the current injection would be another possibility, 
                        however, this is problematic to code since the end of the current injection is not marked.  In the case of the
                        VSD this would not make sense because you would be calculating the voltage at a place where the neuron didn't spike.
                    c.  Set the punishment to some fixed amount that way you can keep track of these episodes.  In this senario you will 
                        the same problems as in b.  However, the added benefit would be that you could track these events for further
                        analysis.  The added drawback is that you would lose the information about how much the spike should be punished.  
                        For example if a model spike occurred at the beginning of the sweep you would think it should be punished more than 
                        at the end of a sweep.
                    Currently I have chosen to do a: throw out all this data    
                '''

        voltage_t0 = self.init_voltage
        threshold_t0 = self.init_threshold
        AScurrents_t0 = self.init_AScurrents

        start_index = 0
        end_index = 0

        try:
            num_spikes = len(bio_spike_time_steps)
            
            # if there are no target spikes, just run until the model spikes
            if num_spikes == 0:  

                start_index = 0
                end_index = len(stim)

                # evaluate the model starting from the beginning until the model spikes
                run_data = self.run_until_biological_spike(voltage_t0, threshold_t0, AScurrents_t0, 
                                                           stim, start_index, end_index, 
                                                           []) 

                voltage = run_data['voltage']
                threshold = run_data['threshold']
                AScurrent_matrix = run_data['AScurrent_matrix']

                if len(voltage)!=len(stim):
                    warnings.warn('YOUR VOLTAGE OUTPUT IS NOT THE SAME LENGTH AS YOUR STIMULUS')
                if len(threshold)!=len(stim):
                    warnings.warn('YOUR THRESHOLD OUTPUT IS NOT THE SAME LENGTH AS YOUR STIMULUS')                
                if len(AScurrent_matrix)!=len(stim):
                    warnings.warn('YOUR AScurrent_matrix OUTPUT IS NOT THE SAME LENGTH AS YOUR STIMULUS')
                                          
                #--right now I am not going to keep track of the spikes in the model that spike if the target doesnt spike.
                #--in essence, I am thowing out the cases when the target neuron doesnt spike
                grid_ISI = np.array([])
                interpolated_ISI = np.array([])

                grid_model_spike_times = np.array([]) #since there is only one spike it is already in reference to stim start
                interpolated_model_spike_times = np.array([]) #since there is only one spike it is already in reference to stim start

                grid_bio_spike_model_voltage = np.array([])
                grid_bio_spike_model_threshold = np.array([])
            else:
                # initialize the output arrays
                grid_ISI = np.empty(num_spikes)
                interpolated_ISI = np.empty(num_spikes)

                grid_model_spike_times = np.empty(num_spikes)
                interpolated_model_spike_times = np.empty(num_spikes) #initialize the array

                grid_bio_spike_model_voltage = np.empty(num_spikes)
                grid_bio_spike_model_threshold = np.empty(num_spikes)

                spikeIndStart = 0
                
                voltage = np.empty(len(stim))
                voltage[:] = np.nan  
                threshold = np.empty(len(stim))
                threshold[:] = np.nan
                AScurrent_matrix = np.empty(shape=(len(stim), len(AScurrents_t0)))
                AScurrent_matrix[:] = np.nan
        
                start_index = 0
                for spike_num in range(num_spikes):
                    if spike_num % 10 == 0:
                        logging.info("spike %d / %d" % (spike_num,  num_spikes))

                    end_index = int(bio_spike_time_steps[spike_num])
    
                    assert start_index < end_index, Exception("start_index > end_index: this is probably because spike_cut_length is longer than the previous inter-spike interval")

                    run_data = self.run_until_biological_spike(voltage_t0, threshold_t0, AScurrents_t0, 
                                                               stim, start_index, end_index, 
                                                               bio_spike_time_steps)

                    voltage[start_index:end_index] = run_data['voltage']
                    threshold[start_index:end_index] = run_data['threshold']
                    AScurrent_matrix[start_index:end_index,:] = run_data['AScurrent_matrix']

                    grid_ISI[spike_num] = run_data['grid_model_spike_time']
                    interpolated_ISI[spike_num] = run_data['interpolated_model_spike_time']

                    grid_model_spike_times[spike_num] = run_data['grid_model_spike_time'] + start_index * self.dt 
                    interpolated_model_spike_times[spike_num] = run_data['interpolated_model_spike_time'] + start_index * self.dt 

                    grid_bio_spike_model_voltage[spike_num] = run_data['grid_bio_spike_model_voltage']
                    grid_bio_spike_model_threshold[spike_num] = run_data['grid_bio_spike_model_threshold']
                    
                    voltage_t0 = run_data['voltage_t0']
                    threshold_t0 = run_data['threshold_t0']
                    AScurrents_t0 = run_data['AScurrents_t0']
                    
                    start_index = end_index

                    if self.spike_cut_length > 0:
                        start_index += self.spike_cut_length
                
                #--get the voltage of the last part of the stim sweep after the last biological spike
                #--currently I am throwing out the data (I am not recording spike times etc) if the model spikes in this time period  
                run_data = self.run_until_biological_spike(voltage_t0, threshold_t0, AScurrents_t0, 
                                                           stim, start_index, len(stim),
                                                           bio_spike_time_steps) #there is no end spike so don't put in a spike

                voltage[start_index:] = run_data['voltage']
                threshold[start_index:] = run_data['threshold']
                AScurrent_matrix[start_index:,:] = run_data['AScurrent_matrix']

            #--the following error functions only work in the case where I am thowing out the sets of data where the target doesnt spike
            if ( len(interpolated_model_spike_times) != num_spikes or 
                 len(grid_model_spike_times) != num_spikes or 
                 len(grid_ISI) != num_spikes or 
                 len(interpolated_ISI) != num_spikes or 
                 len(grid_bio_spike_model_voltage) != num_spikes or 
                 len(grid_bio_spike_model_threshold) != num_spikes):
                raise Exception('The number of spikes in your output does not match your target')

        except GLIFNeuronException, e:
            
            voltage[start_index:end_index] = e.data['voltage']
            threshold[start_index:end_index] = e.data['threshold']
            AScurrent_matrix[start_index:end_index,:] = e.data['AScurrent_matrix']

            out = {
                'voltage': voltage,
                'threshold': threshold,
                'AScurrent_matrix': AScurrent_matrix,

                'grid_ISI': grid_ISI,
                'interpolated_ISI': interpolated_ISI,

                'grid_model_spike_times': grid_model_spike_times,
                'interpolated_model_spike_times': interpolated_model_spike_times,

                'grid_bio_spike_model_voltage': grid_bio_spike_model_voltage,
                'grid_bio_spike_model_threshold': grid_bio_spike_model_threshold
                }

            raise GLIFNeuronException(e.message, out)

        return {
            'voltage': voltage,
            'threshold': threshold,
            'AScurrent_matrix': AScurrent_matrix,

            'grid_model_spike_times': grid_model_spike_times,
            'interpolated_model_spike_times': interpolated_model_spike_times,

            'grid_ISI': grid_ISI,
            'interpolated_ISI': interpolated_ISI,

            'grid_bio_spike_model_voltage': grid_bio_spike_model_voltage,
            'grid_bio_spike_model_threshold': grid_bio_spike_model_threshold
        }
                
    def run_until_biological_spike(self, voltage_t0, threshold_t0, AScurrents_t0, 
                                   stim, start_index, end_index, 
                                   bio_spike_time_steps):

        '''
        Runs The Mihalas Nieber GLIF model for the region in between to biological spikes.  This is used during optimization--
        returned values are used in error functions. The model is optimized in this way to that history effects due to spiking 
        can be adequately optimized.  For example, every time the model spikes a new set of after spike currents will be 
        initiated. To ensure that AScurrents can be optimized we force them to be initiated at the time of the biological spike.
        
        Inputs:
            voltage_t0(scalar): voltage at beginning of simulation (happening between 2 biological spikes)
            threshold_t0(scalar): threshold for a spike at the beginning of the simulation. 
            AScurrents_t0 (array):  values of after spike currents at the beginning of the simulation
            stim (array): stimulus current injected into neuron
            start_index: index of current at which the model should start running.  In practical terms
                this should be at the time of a biological spike + the time that may be cut during spike cutting
            end_index: index of the current at which the model should stop running.  Practically this will be at 
                the next biological spike.
            bio_spike_time_steps: grid time of the biological neuron spike. NOTE: not used here but passed in for the 
                dynamics method.

        Returns a dictionary with:
            voltage: array of voltage
            threshold: array of thereshold 
            AScurrent_matrix: matrix of afterspike currents 

            grid_model_spike_time (float): value at grid precision at which the model spikes
            interpolated_model_spike_time: interpolated value at which model spikes (somewhere between grid spikes) 

            voltage_t0 (float): value of the voltage of the model after it is reset 
            threshold_t0(float): value of the threshold of the model after it is reset
            AScurrents_t0(array): value (amplitude of the afterspike currents after they have been reset (i.e. whatever
                is left over from the last spike added to the currents that will be initiated for the next spike)

            grid_bio_spike_model_voltage (float): voltage of the model at the time of the biological spike (used in VSD)
            grid_bio_spike_model_threshold (float): threshold of the model at the time of biological spike (used in VSD)
           
        '''

        #--preallocate arrays and matricies
        num_time_steps = end_index - start_index
        num_spikes = len(bio_spike_time_steps)

        voltage_out=np.empty(num_time_steps)
        voltage_out[:]=np.nan
        threshold_out=np.empty(num_time_steps)
        threshold_out[:]=np.nan
        AScurrent_matrix=np.empty(shape=(num_time_steps, len(AScurrents_t0)))
        AScurrent_matrix[:]=np.nan

        grid_model_spike_time = None
        interpolated_model_spike_time = None
        
        #--calculate the model values between the two target spikes (don't stop if there is a spike)
        for time_step in xrange(num_time_steps):
            #Note that here you are not recording the first v0 because that was recoded at the end of the previous spike
            voltage_out[time_step]=voltage_t0 
            threshold_out[time_step]=threshold_t0
            AScurrent_matrix[time_step,:]=np.matrix(AScurrents_t0) 
            
            if np.isnan(voltage_t0) or np.isinf(voltage_t0) or np.isnan(threshold_t0) or np.isinf(threshold_t0) or any(np.isnan(AScurrents_t0)) or any(np.isinf(AScurrents_t0)):
                logging.error(self)
                logging.error('time step: %d / %d' % (time_step, num_time_steps))
                logging.error('    voltage_t0: %f' % voltage_t0)
                logging.error('    voltage started the run at: %f' % voltage_out[0])
                logging.error('    voltage before: %s' % voltage_out[time_step-20:time_step])
                logging.error('    threshold_t0: %f' % threshold_t0)
                logging.error('    threshold started the run at: %f' % threshold_out[0])
                logging.error('    threshold before: %s' % threshold_out[time_step-20:time_step])
                logging.error('    AScurrents_t0: %s' % AScurrents_t0)
                raise GLIFNeuronException('Invalid threshold, voltage, or after-spike current encountered.', {
                        'voltage': voltage_out,
                        'threshold': threshold_out,
                        'AScurrent_matrix': AScurrent_matrix
                        })
            
            (voltage_t1, threshold_t1, AScurrents_t1) = self.dynamics(voltage_t0, threshold_t0, AScurrents_t0, stim[time_step+start_index], time_step+start_index, bio_spike_time_steps) #TODO fix list versus array
            
            voltage_t0=voltage_t1
            threshold_t0=threshold_t1
            AScurrents_t0=AScurrents_t1
            
        #figuring out whether model neuron spiked or not        
        for time_step in range(0, num_time_steps): 
            if voltage_out[time_step] > threshold_out[time_step]:
                grid_model_spike_time = self.dt * time_step #not that this should be time_step even though it is index+1 in runModel function because here it isnt recorded until the next step
                interpolated_model_spike_time = interpolate_spike_time(self.dt, time_step-1, threshold_out[time_step-1], threshold_out[time_step], voltage_out[time_step-1], voltage_out[time_step])
                break
                
        # if the last voltage is above threshold and there hasn't already been a spike
        if voltage_t1 > threshold_t1 and grid_model_spike_time is None: 
            grid_model_spike_time = self.dt*num_time_steps
            interpolated_model_spike_time = interpolate_spike_time(self.dt, num_time_steps - 1, threshold_out[num_time_steps-1], threshold_t1, voltage_out[num_time_steps-1], voltage_t1)

        #if the model never spiked, extrapolate to guess when it would have spiked
        if grid_model_spike_time is None: 
            interpolated_model_spike_time = extrapolate_spike_time(self.dt, num_time_steps, threshold_out[-1], threshold_t1, voltage_out[-1], voltage_t1)
            grid_model_spike_time = np.ceil(interpolated_model_spike_time / self.dt) * self.dt  #grid spike time based off extrapolated spike time (changed floor to ceil 5-13-13
        
        # if the target spiked, reset so that next round will start at reset but not recording it in the voltage here.
        # note that at the last section of the stimulus where there is no current injected the model will be reset even if
        # the biological neuron doesn't spike.  However, this doesnt matter as it won't be recorded. 
        if num_spikes > 0:
            (voltage_t0, threshold_t0, AScurrents_t0) = self.reset(voltage_t1, threshold_t1, AScurrents_t1) #reset the variables
        
        return {
            'voltage': voltage_out, 
            'threshold': threshold_out, 
            'AScurrent_matrix': AScurrent_matrix, 

            'grid_model_spike_time': grid_model_spike_time,
            'interpolated_model_spike_time': interpolated_model_spike_time, 

            'voltage_t0': voltage_t0, 
            'threshold_t0': threshold_t0, 
            'AScurrents_t0': AScurrents_t0, 

            'grid_bio_spike_model_voltage': voltage_t1, 
            'grid_bio_spike_model_threshold': threshold_t1            
            }



def extrapolate_spike_time(dt, num_time_steps, threshold_t0, threshold_t1, voltage_t0, voltage_t1):
    return line_crossing_x(dt * num_time_steps, voltage_t0, voltage_t1, threshold_t0, threshold_t1)

def interpolate_spike_time(dt, time_step, threshold_t0, threshold_t1, voltage_t0, voltage_t1):
    return time_step*dt + line_crossing_x(dt, voltage_t0, voltage_t1, threshold_t0, threshold_t1)

def interpolate_spike_value(dt, interpolated_spike_time_offset, v0, v1):
    return v0 + (v1 - v0) * interpolated_spike_time_offset / dt

def line_crossing_x(dx, a0, a1, b0, b1):
    return dx * (b0 - a0) / ( (a1 - a0) - (b1 - b0) )
