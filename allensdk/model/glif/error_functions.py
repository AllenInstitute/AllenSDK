import logging
import sys
import os

from scipy.stats import norm

import numpy as np

from allensdk.model.glif.glif_optimizer_neuron import GlifNeuronException
from allensdk.model.glif.glif_optimizer_neuron import GlifBadInitializationException
from allensdk.model.glif.glif_neuron import GlifBadResetException

import scipy.stats as sps

import numpy as np

# TODO: clean up
# TODO: license
# TODO: remove plotting

'''
NOTE: TRD and TSD were redone here for the current version of the code 
'''

def get_error_function_by_name(func_name):
    if func_name=="TRD":
        return TRD_list_error
    elif func_name=="TSD":
        return square_time_dist_list_error
    elif func_name=="TDD":
        return TDD_list_error
    elif func_name=="VSD":
        return square_voltage_dist_list_error
    elif func_name=="MLIN":
        return MLIN_list_error
    else:
        raise Exception("error function %s does not exist" % func_name)
    
def calculateISIWithSelf(timeStartStim, spikeTimes, time_cut_after_spike=0):
    '''Calculate the interspike interval of a spike train 
    Inputs
        timeStartStim: scalar value time of stimulus onset for the ISI calculation of the ISI of the first spike
        spikeTimes: scalar array of times of spikes
        time_cut_after_spike: often there is cutting happening after the spike when running a model. This is here so that 
        the ISIs can be comparable to the model ISI's
    Output
        ISI: array of scalar values of ISI times
    Note: there is no time being cut on the first spike so we don't do that here
    '''
    if len(spikeTimes)>=2:
        ISItemp=(spikeTimes[1:]-spikeTimes[0:-1])-time_cut_after_spike #calculate the ISI between the spikes in one train
        ISI=np.append(spikeTimes[0]-timeStartStim, ISItemp) #the first ISI is the time of the first spike
    elif len(spikeTimes)==1.0:
        ISI=np.array([spikeTimes[0]-timeStartStim])
    elif len(spikeTimes)==0.0:
        ISI=np.array([])
    return ISI       
    
def calculateISIWRespect2Data(timeStartStim, spikeTimesModel, spikeTimesData):
    '''Calculate the interspike interval of a spike train of model data to the last biological spike
    Inputs
        timeStartStim: scalar value time of stimulus onset for the ISI calculation of the ISI of the first spike
        spikeTimesModel, spikeTimesData: scalar array of times of spikes
    Output
        ISI: array of scalar values of ISI times
    '''
    if len(spikeTimesModel)>=2:
        ISItemp=(spikeTimesModel[1:]-spikeTimesData[0:-1]) #calculate the ISI between the spikes in one train
        ISI=np.append(spikeTimesModel[0]-timeStartStim, ISItemp) #the first ISI is the time of the first spike
    elif len(spikeTimesModel)==1.0:
        ISI=np.array([spikeTimesModel[0]-timeStartStim])
    elif len(spikeTimesModel)==0.0:
        ISI=np.array([])
    return ISI   

def expsymm_pdf(v, sv):
    return 1./(2.*sv)*np.exp(-np.absolute(v)/sv)

def expsymm_cdf(v, sv):
    return 1./2.+(v*(1-np.exp(-np.absolute(v)/sv)))/(2.*np.absolute(v))

def MLIN_list_error(param_guess, experiment, input_data):
    #TODO: binning is now done in preprocessor so perhaps should take it out of here.
    voltage_variance = input_data['subthreshold_long_square_voltage_variance']
#    voltage_distribution=norm(loc=0, scale=np.sqrt(voltage_variance)*10.)
    sv=input_data['sv_for_expsymm'] #used in the expsymm function
#    tau_4AC=experiment.neuron.R_input*experiment.neuron.C
    tau_from_AC=input_data['tau_from_AC']
    spike_length=int(experiment.neuron.spike_cut_length)
    noSpike_bin_size_ind=int(tau_from_AC/experiment.neuron.dt)
    spike_bin_size_time=.005 #TODO: PLAY WITH THIS VALUE. 1 MS, 2, 4, 8
    spike_bin_size_ind=int(spike_bin_size_time/experiment.neuron.dt)

    logging.critical('running parameter guess: %s' % param_guess)
         
    dataSpikeTimes=experiment.grid_spike_times   

    MLIN_list = []
    try:
        run_data = experiment.run(param_guess)

    except GLIFNeuronException, e:
        out=e.data
        raise e
        modelSpikeISI=[e.data['interpolated_ISI']]    
        
    except GLIFBadInitializationException, e:
        logging.critical('voltage STARTS above threshold: setting error to be large.Difference between thresh and voltage is: %f' % e.dv)
        raise Exception()
    except GLIFBadResetException, e:
        logging.critical('THIS REALLY SHOULDNT HAPPEN WITH NEW INITIALIZATION EXCEPTION: voltage is above threshold at reset: setting error to be large.  Difference between thresh and voltage is: %f' % e.dv)
        raise Exception()

    v_model_list=[]
    th_model_list=[] 
    non_spike_bin_ind_edges_list=[]
    spike_edges_list_list=[]
    spike_bins_list=[]
    noSpike_bins_list=[]
    spike_prob_list=[]
    noSpike_prob_list=[]

    for stim_list_index in range(0,len(experiment.stim_list)):
    #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if len(experiment.spike_time_steps[stim_list_index])==0:
            MLIN=0
            raise Exception('there are no spikes in the sweep')
        else:
            bio_spike_ind=experiment.spike_time_steps[stim_list_index]
            v_model=run_data['voltage'][stim_list_index]
            th_model=run_data['threshold'][stim_list_index]
            #------------------------------------------------------------------------------------
            #---------------make all the bins----------------------------------------------------
            #------------------------------------------------------------------------------------
            
            #--for every spike define a region for making non spiking bins
            between_spike_edges_list=[]
            between_spike_edges_list.append([0, bio_spike_ind[0]-spike_bin_size_ind])  #edges to first spike
            for ii in range(0, len(bio_spike_ind)-1):
                between_spike_edges_list.append([bio_spike_ind[ii]+spike_length, bio_spike_ind[ii+1]-spike_bin_size_ind])
            
            
            #--define no spike bin edges
            non_spike_bin_ind_edges_in_ISI=[]
            for btw_spike_edges in between_spike_edges_list:
                temp=range(btw_spike_edges[0], btw_spike_edges[1], noSpike_bin_size_ind)
                temp.append(btw_spike_edges[1])
                non_spike_bin_ind_edges_in_ISI.append(temp)
            
            #--define spike bin edges
            spike_edges_list=[]
            for spike in bio_spike_ind:
                spike_edges_list.append([spike-spike_bin_size_ind, spike])
            
            #--nonspike bin edges need to be arranged correctly because they are just all the edges for a given ISI
            non_spike_bin_ind_edges=[]
            for edges_in_1_spike in non_spike_bin_ind_edges_in_ISI:
                for ii in range(0,len(edges_in_1_spike)-1):
                    non_spike_bin_ind_edges.append([edges_in_1_spike[ii], edges_in_1_spike[ii+1]])
            
            #-----------------------------------------------------------------------------
            #-------------finding values in bins------------------------------------------
            #-----------------------------------------------------------------------------
            
            spike_bins={}
#            spike_bins['vmax']=[]
            spike_bins['th']=[]
            spike_bins['v']=[]
            spike_bins['v_th_diff']=[]
#            spike_bins['ind_of_max_v']=[]
            spike_bins['ind_of_max_diff_btw_v_th']=[]
            
            for bin_edges in spike_edges_list:
                bin_ind=range(bin_edges[0], bin_edges[1])
                #vmax_in_bin=max(v_model[bin_ind])
                diff_vector=th_model[bin_ind]-v_model[bin_ind]
                #the_ind=bin_ind[np.where(v_model[bin_ind]==vmax_in_bin)[0]] this is used to use in where v is at a max (as opposed to the difference between v and th)
#                print "***********************************************************"
#                print 'th_model[bin_ind]', th_model[bin_ind]
#                print 'v_model[bin_ind]',v_model[bin_ind]
#                print 'diff_vector', diff_vector
#                print 'np.where(diff_vector==min(diff_vector))[0]', np.where(diff_vector==min(diff_vector))[0]
                the_ind=bin_ind[np.where(diff_vector==min(diff_vector[~np.isnan(diff_vector)]))[0][0]]
                th_in_bin=th_model[the_ind]
                v_in_bin=v_model[the_ind]
#                diffV=th_in_bin-vmax_in_bin
                diffV=th_model[the_ind]-v_model[the_ind]
                spike_bins['v'].append(v_in_bin)
                spike_bins['th'].append(th_in_bin)
                spike_bins['v_th_diff'].append(diffV)
                spike_bins['ind_of_max_diff_btw_v_th'].append(the_ind)
            
            
            noSpike_bins={}
#            noSpike_bins['vmax']=[]
            noSpike_bins['th']=[]
            noSpike_bins['v']=[]
            noSpike_bins['v_th_diff']=[]
            noSpike_bins['ind_of_max_diff_btw_v_th']=[]
            for bin_edges in non_spike_bin_ind_edges:
                bin_ind=range(bin_edges[0], bin_edges[1])
#                vmax_in_bin=max(v_model[bin_ind])
                diff_vector=th_model[bin_ind]-v_model[bin_ind]
#                max_indicies=np.where(v_model[bin_ind]==vmax_in_bin)[0]
                min_indicies=np.where(diff_vector==min(diff_vector))[0]
#                if len(max_indicies)>1:
#                    print 'there is more than one maximum indicie in a bin at', max_indicies, 'choosing last value for computation'
#                    print 'all voltages in the bin are', v_model[bin_ind]
                the_ind=bin_ind[min_indicies[-1]] #this is here just incase there is more than one value at max voltage in a bin
                th_in_bin=th_model[the_ind]
                v_in_bin=v_model[the_ind]
                diffV=th_model[the_ind]-v_model[the_ind]
                noSpike_bins['v'].append(v_in_bin)
                noSpike_bins['th'].append(th_in_bin)
                noSpike_bins['v_th_diff'].append(diffV)
                noSpike_bins['ind_of_max_diff_btw_v_th'].append(the_ind)

    #-----------------------------------------------------------------------------
    #-------------calculate MLIN--------------------------------------------------
    #-----------------------------------------------------------------------------

#    this was the version with the normal distribution
#             noSpike_prob=np.log(np.spacing(1)+voltage_distribution.cdf(noSpike_bins['v_th_diff']))
#             spike_prob=np.log(1+np.spacing(1)-voltage_distribution.cdf(spike_bins['v_th_diff']))
            #version with the expsymm function
            
        #---OPTION ONE--------
#            noSpike_prob=np.log(np.spacing(1)+expsymm_cdf(noSpike_bins['v_th_diff'], sv))
#            spike_prob=np.log(1+np.spacing(1)-expsymm_cdf(spike_bins['v_th_diff'], sv))
#        #---OPTION TWO--------
#            N_spike=np.float(len(spike_bins['v_th_diff']))
#            N_noSpike=np.float(len(noSpike_bins['v_th_diff']))
#            noSpike_prob=(N_spike/(N_noSpike+N_spike))*np.log(np.spacing(1)+expsymm_cdf(noSpike_bins['v_th_diff'], sv))
#            spike_prob=(N_noSpike/(N_noSpike+N_spike))*np.log(1+np.spacing(1)-expsymm_cdf(spike_bins['v_th_diff'], sv))       
        #---OPTION THREE
            noSpike_negDiff=(-np.log(2.)+np.array(noSpike_bins['v_th_diff'])/sv)[np.array(noSpike_bins['v_th_diff'])<=0.0]
            noSpike_posDiff=(np.log(1.-0.5*np.exp(-np.array(noSpike_bins['v_th_diff'])/sv)))[np.array(noSpike_bins['v_th_diff'])>0.0]
            noSpike_prob=np.append(noSpike_negDiff, noSpike_posDiff)  #!!NOTE: this may not line up correctly in outputs of MLIN HACK
            
            spike_negDiff=(np.log(1.-0.5*np.exp(np.array(spike_bins['v_th_diff'])/sv)))[np.array(spike_bins['v_th_diff'])<=0.0]
            spike_posDiff=(-np.log(2.)-np.array(spike_bins['v_th_diff'])/sv)[np.array(spike_bins['v_th_diff'])>0.0]
            spike_prob=np.append(spike_negDiff, spike_posDiff)
            
            MLIN=-(sum(noSpike_prob)+sum(spike_prob))            
            print 'MLIN', MLIN
                
        MLIN_list.append([MLIN])
        v_model_list.append(v_model)
        th_model_list.append(th_model)
        non_spike_bin_ind_edges_list.append(non_spike_bin_ind_edges) 
        spike_edges_list_list.append(spike_edges_list)
        spike_bins_list.append(spike_bins)
        noSpike_bins_list.append(noSpike_bins)
        spike_prob_list.append(spike_prob)
        noSpike_prob_list.append(noSpike_prob)

    concatenateMLINList=np.concatenate(MLIN_list)
    experiment.spike_errors.append(concatenateMLINList)
          
#    print 'param Guess', param_guess, 'TRD', np.mean(concatenateTRDList)
    out =np.mean(concatenateMLINList) 
    logging.critical('MLIN: %f' % np.mean(concatenateMLINList))

#-------------------------------------------------------------------
#--------------------------plotting------------------------------------
#---------------------------------------------------------------------
    time=np.arange(0, len(v_model))*experiment.neuron.dt    
#    plt.subplot(2,1,1)
#    plt.title('Model', fontsize=16)
#    plt.plot(time, v_model, 'b-', label='voltage')
#    plt.plot(time, th_model, 'b--', label='threshold')
#    plt.plot(np.concatenate(non_spike_bin_ind_edges)*experiment.neuron.dt, v_model[np.concatenate(non_spike_bin_ind_edges)], 'k|', ms=16)
#    plt.plot(np.concatenate(spike_edges_list)*experiment.neuron.dt, v_model[np.concatenate(spike_edges_list)], 'r|', ms=16)
#    plt.plot(np.array(spike_bins['ind_of_max_diff_btw_v_th'])*experiment.neuron.dt, v_model[spike_bins['ind_of_max_diff_btw_v_th']], 'r.', ms=6)
#    plt.plot(np.array(spike_bins['ind_of_max_diff_btw_v_th'])*experiment.neuron.dt, th_model[spike_bins['ind_of_max_diff_btw_v_th']], 'r.', ms=6)
#    plt.plot(np.array(noSpike_bins['ind_of_max_diff_btw_v_th'])*experiment.neuron.dt, v_model[noSpike_bins['ind_of_max_diff_btw_v_th']], 'k.', ms=6)
#    plt.plot(np.array(noSpike_bins['ind_of_max_diff_btw_v_th'])*experiment.neuron.dt, th_model[noSpike_bins['ind_of_max_diff_btw_v_th']], '.', ms=6)
#    plt.title(' MLIN='+ str(MLIN)+' : sv='+str(sv)+' : ac_tau='+str(tau_from_AC)+' : spike bin size='+str(spike_bin_size_time)+'!!!!!! !!!!!', fontsize=20)
#    plt.xlim([0, time[-1]])
#
#    plt.subplot(2,1,2)
#    plt.plot(np.array(spike_bins['ind_of_max_diff_btw_v_th'])*experiment.neuron.dt, spike_prob, 'r.', ms=16, label='spike probablility')
#    plt.plot(np.array(noSpike_bins['ind_of_max_diff_btw_v_th'])*experiment.neuron.dt, noSpike_prob, 'b.', ms=16, label='no spike probablility')
#    plt.legend()
#
#    print "coming out of function", out
#
#    plt.show()


    #converted to list 4_11_15
    experiment.MLIN_HACK={
        'v_model': v_model_list,
        'th_model': th_model_list,
        'non_spike_bin_ind_edges': non_spike_bin_ind_edges_list,
        'spike_edges_list': spike_edges_list_list, #TODO Corinne: why was this originally a list but nothing else a list
        'spike_bins': spike_bins_list,
        'noSpike_bins': noSpike_bins_list,
        'tau_from_AC': tau_from_AC,
        'spike_prob': spike_prob_list,
        'noSpike_prob': noSpike_prob_list,
        'spike_bin_size_time' : spike_bin_size_time,
        'sv': sv
        
    }
#    
    return out

    

def TRD_list_error(param_guess, experiment, input_data):
    '''
    gets called from the optimizer once each optimizer iteration.
    TRDerror iterates over the stim_list list of stimulus vectors and passes them to the neuron run method
    @param param_guess: a vector of scalar parameters
    @param experiment: a neuron experiment wrapping the neuron class
    @return: a scalar representing the error over all items in the stimulus list
    @note: the neuron experiment must have a stim_list member
    '''   
    logging.critical('running parameter guess: %s' % param_guess)
         
    dataSpikeTimes=experiment.grid_spike_times

    TRD_list = []
    try:
        run_data = experiment.run(param_guess)
        modelSpikeISI=run_data['interpolated_ISI']
#         print modelSpikeISI

    except GLIFNeuronException, e:
        out=e.data
        raise e
        modelSpikeISI=[e.data['interpolated_ISI']]
    except GLIFBadInitializationException, e:
        logging.critical('voltage is above threshold setting error to be large.  Difference between thresh and voltage is: %f' % e.dv)
        raise Exception()
   
    for stim_list_index in range(0,len(experiment.stim_list)):
        #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if len(experiment.spike_time_steps[stim_list_index])==0:
            TRDout=[0]
        else:
            #bug found 5-16-13: TRD was being calculated in terms of stimulus spike time instead of ISI type spike time.  See beloww            
            #TRDout=TRD(experiment.interpolated_spike_time_target_list[stim_list_index], spikeActualTime_list[stim_list_index])#BUGGY!
            '''TODO: THE NAME OF THIS SCKETCHES ME OUT.  I THINK IT SHOULD BE CALCULATING THE TIMES SO THE ISI CAN BE CALCULATED BELOW'''
            #TODO: why am I using grid and interpolated here
            #print "SUSPECT", dataSpikeTimes[stim_list_index]
            ISITarget = calculateISIWithSelf(0, dataSpikeTimes[stim_list_index], experiment.neuron.spike_cut_length*experiment.dt)
#            print '-----IN TRD FUNCTION-------'
            #print 'ISITarget', ISITarget
            #print 'modelSpikeISI', modelSpikeISI[stim_list_index]
#            print 'diff between ISITarget and ISI Model from last target spike', ISITarget-gridISIFromLastTargSpike_list[stim_list_index]
            TRDout=TRD(ISITarget, modelSpikeISI[stim_list_index])            
        
        TRD_list.append(TRDout) 

#    print 'TRD_list', TRD_list
    concatenateTRDList=np.concatenate(TRD_list)
    experiment.spike_errors=TRD_list
          
#    print 'param Guess', param_guess, 'TRD', np.mean(concatenateTRDList)
    out =np.mean(concatenateTRDList) 
    logging.critical('TRD: %f' % np.mean(concatenateTRDList))

    return out

def TDD_list_error(param_guess, experiment, input_data):
    '''
    gets called from the optimizer once each optimizer iteration.
    TRDerror iterates over the stim_list list of stimulus vectors and passes them to the neuron run method
    @param param_guess: a vector of scalar parameters
    @param experiment: a neuron experiment wrapping the neuron class
    @return: a scalar representing the error over all items in the stimulus list
    @note: the neuron experiment must have a stim_list member
    '''   
    logging.critical('running parameter guess: %s' % param_guess)
         
    dataSpikeTimes=experiment.grid_spike_times

    TDD_list = []
    try:
        run_data = experiment.run(param_guess)
        modelSpikeISI=run_data['interpolated_ISI']

    except GLIFNeuronException, e:
        out=e.data
        raise e
        modelSpikeISI=[e.data['interpolated_ISI']]
    except GLIFBadInitializationException, e:
        logging.critical('voltage is above threshold setting error to be large.  Difference between thresh and voltage is: %f' % e.dv)
        raise Exception()
   
    for stim_list_index in range(0,len(experiment.stim_list)):
        #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if len(experiment.spike_time_steps[stim_list_index])==0:
            TDDout=[0]
        else:
            #bug found 5-16-13: TRD was being calculated in terms of stimulus spike time instead of ISI type spike time.  See beloww            
            #TRDout=TRD(experiment.interpolated_spike_time_target_list[stim_list_index], spikeActualTime_list[stim_list_index])#BUGGY!
            '''TODO: THE NAME OF THIS SCKETCHES ME OUT.  I THINK IT SHOULD BE CALCULATING THE TIMES SO THE ISI CAN BE CALCULATED BELOW'''
            #TODO: why am I using grid and interpolated here
            #print "SUSPECT", dataSpikeTimes[stim_list_index]
            ISITarget = calculateISIWithSelf(0, dataSpikeTimes[stim_list_index], experiment.neuron.spike_cut_length*experiment.dt)


#             print 'COMPUTEING'
#            print '-----IN TRD FUNCTION-------'
            #print 'ISITarget', ISITarget
            #print 'modelSpikeISI', modelSpikeISI[stim_list_index]
#            print 'diff between ISITarget and ISI Model from last target spike', ISITarget-gridISIFromLastTargSpike_list[stim_list_index]
            TDDout=TDD(ISITarget, modelSpikeISI[stim_list_index])            
        
        TDD_list.append(TDDout) 

    concatenateTDDList=np.concatenate(TDD_list)
    experiment.spike_errors=TDD_list
    out =np.mean(concatenateTDDList) 
    logging.critical('TDD: %f' % out)

    return out

# def TRD(data, model):
#    """This code is not right, but here for posterity"""
#
#
#     data = data.astype(np.float64)
#     model = model.astype(np.float64)
#     
#     return (1-(model/data)**np.sign(data-model))**2.

def TDD(target_isi_array, obtained_isi_array):
#     print obtained_isi_array
    
    f = lambda x:x**2
#     f = lambda x:x**2/(1+x**2)
    
    shape, loc, scale = sps.gamma.fit(target_isi_array)
    best_fit_rv = sps.gamma(a=shape, loc=loc, scale=scale)
    result_list = []
#     print result_list
    for d, m in zip(target_isi_array, obtained_isi_array):
        
        if not (np.isnan(best_fit_rv.cdf(d)) or np.isnan(best_fit_rv.cdf(m))):
            result_list.append( f(best_fit_rv.cdf(d) - best_fit_rv.cdf(m)))
    
#     print result_list
    return result_list

def TRD(tSpikeTargetISI, tSpikeObtainedISI):
#     print tSpikeObtainedISI
#Given two arrays of spike times (as defined as the time (ISI proxy) from the last target spike)
# calculate the time ratio distance as defined in Stefan's draft    
#    inputs
#        tSpikeTargetISI & tSpikeObtainedISI: arrays with time values (in reference to last target spike)
#        of spikes in the trains
#    outputs
#        out: values of TRD for every set of spikes
       
    if len(tSpikeTargetISI)!=len(tSpikeObtainedISI): 
        #!!!!!tSpikeObtainedISI is not correct here)!!!!!!!
        #given the structure of optimization problem every target (biological) spike
        #should have a corresponding model spike--if not, something is wrong
        print 'lengths of vector are', len(tSpikeTargetISI), 'vs', len(tSpikeObtainedISI)
        raise Exception('The two time vectors for the TRD are not the same length')
    
    out=np.zeros(len(tSpikeObtainedISI)) #initiate vector
    for i in range(0,len(tSpikeObtainedISI)):
        if tSpikeObtainedISI[i]>=0 and tSpikeObtainedISI[i]<=tSpikeTargetISI[i]: #if model spike bigger than zero and happens before bio spike
            out[i]=(1-(tSpikeObtainedISI[i]/tSpikeTargetISI[i]))**2
        elif tSpikeObtainedISI[i]<0:
            out[i]=(1-(tSpikeTargetISI[i]/tSpikeObtainedISI[i]))**2
        elif tSpikeObtainedISI[i]>tSpikeTargetISI[i]: #if model spike happens after bio spike
            out[i]=(1-(tSpikeTargetISI[i]/tSpikeObtainedISI[i]))**2
        else:
            raise Exception("TRD is undefined")
        
    return out



def square_time_dist_list_error(param_guess, experiment, input_data):
    '''
    square_time_dist_list_error gets called from the optimizer once each optimizer iteration.
    It iterates over the stim_list list of stimulus vectors and passes them to the neuron run method.
    inputs:
        param_guess: a vector of scalar parameters.
        experiment: a neuron experiment wrapping the neuron class.
    returns 
    a scalar representing the error over all items in the stimulus list
    @note: the neuron experiment must have a stim_list member
    ''' 
    logging.critical('running parameter guess: %s' % param_guess)

    dataSpikeTimes=experiment.grid_spike_times

    TSD_list = []
    try:
        run_data = experiment.run(param_guess)
        modelSpikeISI=run_data['interpolated_ISI']

    except GLIFNeuronException, e:
        out=e.data
        raise e
        modelSpikeISI=[e.data['interpolated_ISI']]
    except GLIFBadResetException, e:
        print e.dv
        raise e
    
    for stim_list_index in range(0,len(experiment.stim_list)):
    #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if len(experiment.spike_time_steps[stim_list_index])==0:
            TSDout=[0]
        else:
            ISITarget = calculateISIWithSelf(0, dataSpikeTimes[stim_list_index], experiment.neuron.spike_cut_length*experiment.dt)
#            print '-----IN TRD FUNCTION-------'
#            print 'ISITarget', ISITarget
#            print 'gridISIFromLastTargSpike_list', gridISIFromLastTargSpike_list[stim_list_index]
#            print 'diff between ISITarget and ISI Model from last target spike', ISITarget-gridISIFromLastTargSpike_list[stim_list_index]
            TSDout = squareTimeDist(ISITarget, modelSpikeISI[stim_list_index])

            
        TSD_list.append(TSDout) 
        
        
        
#    print 'SD_list', SD_list
    concatenateTSDList=np.concatenate(TSD_list)
    experiment.spike_errors.append(concatenateTSDList)

    logging.critical('TSD: %f' % np.mean(concatenateTSDList))    
    return np.mean(concatenateTSDList) 

def squareTimeDist(tSpikeTargetISI, tSpikeObtainedISI):  
#Given two arrays of spike times (as defined as the time (ISI proxy) from the last target spike)
# calculate the square distance error.   
#    inputs
#        tSpikeTargetISI & tSpikeObtainedISI: arrays with time values (in reference to last target spike)
#        of spikes in the trains
#    outputs
#        out: square distance values for every set of spikes
       #PBS -N Example_run

    if len(tSpikeTargetISI)!=len(tSpikeObtainedISI): 
        #!!!!!tSpikeObtainedISI is not correct here)!!!!!!!
        #given the structure of optimization problem every target (biological) spike
        #should have a corresponding model spike--if not, something is wrong
        raise Exception('The two time vectors for the TSD are not the same length')
    
    out=np.zeros(len(tSpikeObtainedISI)) #initiate vector
    for i in range(0,len(tSpikeObtainedISI)):
        out[i]=((tSpikeObtainedISI[i]-tSpikeTargetISI[i])/tSpikeTargetISI[i])**2

    return out

def square_voltage_dist_list_error(param_guess, experiment, input_data):
    '''
    squareDistListError gets called from the optimizer once each optimizer iteration.
    It iterates over the stim_list list of stimulus vectors and passes them to the neuron run method.
    inputs:
        param_guess: a vector of scalar parameters.
        experiment: a neuron experiment wrapping the neuron class.
    returns: 
        a scalar representing the error over all items in the stimulus list
    @note: the neuron experiment must have a stim_list member
    ''' 

    logging.critical('running parameter guess: %s' % param_guess)

    VSD_list = []

    run_data = experiment.run(param_guess)
    
    
    VSD_list = []
    try:
        run_data = experiment.run(param_guess)
        modelSpikeISI=run_data['interpolated_ISI']

    except GLIFNeuronException, e:
        out=e.data
        raise e
        modelSpikeISI=[e.data['interpolated_ISI']]    
    except GLIFBadResetException, e:
        print e.dv
        raise e

  
 # Print things out here
  
    for stim_list_index in range(0,len(experiment.stim_list)):
    #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if (len(experiment.spike_time_steps[stim_list_index]) == 0 or experiment.target_spike_mask[stim_list_index] == False):
            VSDout = [0]
        else:
            print run_data['grid_bio_spike_model_voltage'][stim_list_index]
            print run_data['grid_bio_spike_model_threshold'][stim_list_index]
            VSDout = square_voltage_dist(run_data['grid_bio_spike_model_voltage'][stim_list_index], 
                                         run_data['grid_bio_spike_model_threshold'][stim_list_index], 
                                         experiment)
        VSD_list.append(VSDout) 

#    print 'VSD_list', VSD_list
    concatenateVSDList=np.concatenate(VSD_list)
    experiment.spike_errors.append(concatenateVSDList)

    logging.critical('VSD: %f' % np.mean(concatenateVSDList))

#    if fileName!=None:
#        print 'file name ins vsd is', fileName
#        with open(os.path.splitext(fileName)[0]+'VSD.txt', 'a+') as f:
#            np.savetxt(f, np.mean(concatenateVSDList))

#--------Put this in if you want to do diagnostic plots but you can only be doing one sweep
#    diagnostic_plot_one_sweep(voltage_list, threshold_list, AScurrentMatrix_list, modelGridSpikeTime_list, 
#                           modelSpikeInterpolatedTime_list, gridISIFromLastTargSpike_list, interpolatedISIFromLastTargSpike_list, 
#                           voltageOfModelAtGridBioSpike_list, threshOfModelAtGridBioSpike_list, voltageOfModelAtInterpolatedBioSpike_list, 
#                           thresholdOfModelAtInterpolatedBioSpike_list, 
#                           experiment.grid_spike_time_target_list, experiment.neuron, experiment.stim_list, VSD_list, np.mean(concatenateVSDList))
#    #get rid of this when actually doing optimization because it is costly
#    #TODO: IF YOU WANT TO INCORPORATE IT IN THE FUTURE YOU NEED TO BUILD INTO THE OPTIMIZER SO CREATE NEW DIRECTIORIES FOR EACH NEW FITNEURON START UP
#
#    #have to annotate out here since there is no knowledge of the the parameters inside
#    plt.annotate(str(experiment.fit_names_list)+ str(param_guess), 
#             xy=(.05, .975),
#             xycoords='figure fraction',
#             horizontalalignment='left', verticalalignment='top',
#             fontsize=20)
#    
#    
#    if fileName!=None:
#        directory=os.path.splitext(fileName)[0]+'/VSDprogressionPlots'
#        if not os.path.exists(directory):
#            os.makedirs(directory)
#        filesInThere=sort(os.listdir(directory)) #get all files currently in the directory
#        if len(filesInThere)==0:
#            plt.savefig(os.path.join(directory, 'iter_0000.png'), format='png')
#        else:
#            lastFileNum=os.path.splitext(filesInThere[-1])[0][-4:]
#            plt.savefig(os.path.join(directory, 'iter_'+str(int(lastFileNum)+1).zfill(4)+'.png'), format='png')
#    #plt.show()
#    plt.close()
       
    return np.mean(concatenateVSDList) 

def square_voltage_dist(voltageOfModelAtBioSpike_array, threshOfModelAtBioSpike_array, experiment):
    '''
    inputs
        voltageOfModelAtBioSpike_array and threshOfModelAtBioSpike_array: arrays of thresholds and voltage of model at every target (bio) spike
        experiment: a neuron experiment wrapping the neuron class.
    outputs
        out: sqared difference between voltage and threshold of model neuron at each target (bio) spike'''
       
    if len(voltageOfModelAtBioSpike_array)!=len(threshOfModelAtBioSpike_array): 
        raise Exception('The two time vectors for the VSD are not the same length')
    
    out=np.zeros(len(voltageOfModelAtBioSpike_array)) #initiate vector
    for i in range(0,len(voltageOfModelAtBioSpike_array)):
        out[i]=((threshOfModelAtBioSpike_array[i]-voltageOfModelAtBioSpike_array[i])/(experiment.neuron.th_inf-experiment.neuron.El))**2

    return out

def diagnostic_plot_one_sweep(BP_v_list, BP_thresh_list, BP_AScurrentMatrix_list, BP_modelGridSpikeTime_list, 
                   BP_modelSpikeInterpolatedTime_list, BP_gridISIFromLastTargSpike_list, BP_interpolatedISIFromLastTargSpike_list, 
                   BP_v_OfModelAtGridBioSpike_list, BP_thresh_OfModelAtGridBioSpike_list, BP_v_OfModelAtInterpolatedBioSpike_list, 
                   BP_threshOfModelAtInterpolatedBioSpike_list, 
                   grid_spike_time_target_list, the_neuron, stim_list, VSD_list, totalVSD): 
    '''this is a huge hack to look at stuff.  You can only do it with a single sweep comming though'''
    if len(stim_list)>1:
        raise Exception()
    if len(grid_spike_time_target_list[0])!=len(BP_modelGridSpikeTime_list[0]):
        raise Exception('your bio spikes and model spikes are not the same number')
    BP_time=np.arange(0,len(BP_v_list[0]))*the_neuron.dt
    plt.figure(figsize=(15,11))
    plt.subplot(4,1,1)
    plt.plot(BP_time,stim_list[0])
    plt.title('current injection')
    plt.subplot(4,1,2)
    plt.plot(BP_time,BP_v_list[0], label='model voltage',  lw=2)
    plt.plot(BP_time, BP_thresh_list[0], label='model threshold',  lw=2)
    plt.plot(grid_spike_time_target_list[0], np.ones(len(BP_modelGridSpikeTime_list[0]))*the_neuron.th_inf, label='thr_inf', lw=2)

    plt.plot(grid_spike_time_target_list[0], BP_v_OfModelAtGridBioSpike_list[0], 'xk', ms=14, lw=2)
    plt.plot(grid_spike_time_target_list[0], BP_thresh_OfModelAtGridBioSpike_list[0], 'ok', ms=14)
    #plot where bio spike was and where it ended up but only needed for the TRD
#    for jj in range(0, len(grid_spike_time_target_list[0])):
#        #lines for voltage at time of bio spike and at interpolated model spike
#        plt.plot([grid_spike_time_target_list[0][jj], grid_spike_time_target_list[0][jj]+BP_interpolatedISIFromLastTargSpike_list[0][jj]],
#                  [BP_v_OfModelAtGridBioSpike_list[0][jj], BP_v_OfModelAtInterpolatedBioSpike_list[0][jj]], 'c-',  lw=2)
#        #lines for threshold at time of bio spike and at interpolated model spike
#        plt.plot([grid_spike_time_target_list[0][jj], grid_spike_time_target_list[0][jj]+BP_interpolatedISIFromLastTargSpike_list[0][jj]],
#                  [BP_thresh_OfModelAtGridBioSpike_list[0][jj], BP_threshOfModelAtInterpolatedBioSpike_list[0][jj]], 'm-',  lw=2)
    plt.title('x and o are voltage and threshold at biospike')
    plt.legend()    
    print 'thr_inf', the_neuron.th_inf
    print 'the_neuron.Vr', the_neuron.Vr, 'the_neuron.v_reset_method', the_neuron.v_reset_method, 'the_neuron.vthresh_reset_method', the_neuron.vthresh_reset_method

    plt.subplot(4,1,3)
    plt.bar(range(len(VSD_list[0])), VSD_list[0])
    plt.title('VSD='+str(totalVSD))
    plt.subplot(4,1,4)
    for ii in range(0, BP_AScurrentMatrix_list[0].shape[1]):
        plt.plot(BP_time, BP_AScurrentMatrix_list[0][:,ii], lw=2)
    plt.title('AfterSpike Currents')
    plt.tight_layout()

