import logging

import matplotlib.pyplot as plt
import sys
#from spikeTimeDiffBtwObtainAndTarg import *
from scipy.optimize import fmin
from time import time
from numpy import ones, zeros, concatenate, mean, savetxt, arange, sort
import os

import numpy as np

def get_error_function_by_name(func_name):
    if func_name=="TRD":
        return TRD_list_error
    if func_name=="TSD":
        return square_time_dist_list_error
    if func_name=="VSD":
        return square_voltage_dist_list_error

def TRD_list_error(param_guess, experiment):
    '''
    gets called from the optimizer once each optimizer iteration.
    TRDerror iterates over the stim_list list of stimulus vectors and passes them to the neuron run method
    @param param_guess: a vector of scalar parameters
    @param experiment: a neuron experiment wrapping the neuron class
    @return: a scalar representing the error over all items in the stimulus list
    @note: the neuron experiment must have a stim_list member
    '''        
    TRD_list = []

    run_data = experiment.run(param_guess)
    print "run_data['interpolated_spike_times']", run_data['interpolated_spike_times']
    print "experiment.interpolated_spike_times", experiment.interpolated_spike_times
  
    for stim_list_index in range(0,len(experiment.stim_list)):
    #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if len(experiment.spike_time_steps[stim_list_index])==0:
            TRDout=[0]
        else:
            #bug found 5-16-13: TRD was being calculated in terms of stimulus spike time instead of ISI type spike time.  See beloww            
            #TRDout=TRD(experiment.interpolated_spike_time_target_list[stim_list_index], spikeActualTime_list[stim_list_index])#BUGGY!
            ISITarget=calculateISIFromIntTime(0, experiment.interpolated_spike_times[stim_list_index], experiment.grid_spike_times[stim_list_index])
            
#            print '-----IN TRD FUNCTION-------'
#            print 'ISITarget', ISITarget
#            print 'gridISIFromLastTargSpike_list', gridISIFromLastTargSpike_list[stim_list_index]
#            print 'diff between ISITarget and ISI Model from last target spike', ISITarget-gridISIFromLastTargSpike_list[stim_list_index]
            TRDout=TRD(ISITarget, run_data['interpolated_ISI'][stim_list_index])
        TRD_list.append(TRDout) 
#    print 'TRD_list', TRD_list
    concatenateTRDList=concatenate(TRD_list)          
#    print 'param Guess', param_guess, 'TRD', mean(concatenateTRDList)
    out =mean(concatenateTRDList) 
#    print 'param_guess', param_guess
#    print 'mean(concatenateTRDList): ', out
    return out

def TRD(tSpikeTargetISI, tSpikeObtainedISI):
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
        raise Exception('The two time vectors for the TRD are not the same length')
    
    out=zeros(len(tSpikeObtainedISI)) #initiate vector
    for i in range(0,len(tSpikeObtainedISI)):
        if tSpikeObtainedISI[i]>=0 and tSpikeObtainedISI[i]<tSpikeTargetISI[i]: #I am not sure why the >=0 requirement is there yet
            out[i]=(1-(tSpikeObtainedISI[i]/tSpikeTargetISI[i]))**2
        else:
            out[i]=(1-(tSpikeTargetISI[i]/tSpikeObtainedISI[i]))**2

    return out

def square_time_dist_list_error(param_guess, experiment):
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
    TSD_list = []   
    run_data = experiment.run(param_guess)

    for stim_list_index in range(0,len(experiment.stim_list)):
    #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if len(experiment.spike_time_steps[stim_list_index])==0:
            TSDout=[0]
        else:
            ISITarget = calculateISIFromIntTime(0, experiment.interpolated_spike_times[stim_list_index], experiment.grid_spike_times[stim_list_index])
#            print '-----IN TRD FUNCTION-------'
#            print 'ISITarget', ISITarget
#            print 'gridISIFromLastTargSpike_list', gridISIFromLastTargSpike_list[stim_list_index]
#            print 'diff between ISITarget and ISI Model from last target spike', ISITarget-gridISIFromLastTargSpike_list[stim_list_index]
            TSDout = squareTimeDist(ISITarget, run_data['interpolated_ISI'][stim_list_index])
        TSD_list.append(TSDout) 
#    print 'SD_list', SD_list
    concatenateTSDList=concatenate(TSD_list)    
    return mean(concatenateTSDList) 

def squareTimeDist(tSpikeTargetISI, tSpikeObtainedISI):  #FOR THIS I NEED THE VOLTAGE AND THE THRESHOLD. SO RUN UNTIL SPIKE WILL HAVE TO SPIT OUT THE THRESH AND VOLT VALUES
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
    
    out=zeros(len(tSpikeObtainedISI)) #initiate vector
    for i in range(0,len(tSpikeObtainedISI)):
        out[i]=((tSpikeObtainedISI[i]-tSpikeTargetISI[i])/tSpikeTargetISI[i])**2

    return out

def square_voltage_dist_list_error(param_guess, experiment):
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

    logging.info('running parameter guess: %s' % param_guess)

    VSD_list = []

    run_data = experiment.run(param_guess)
  
 # Print things out here
  
    for stim_list_index in range(0,len(experiment.stim_list)):
    #TODO: the following line is a hack to take care of the case when there are no spikes in a sweep
        if (len(experiment.spike_time_steps[stim_list_index]) == 0 or experiment.target_spike_mask[stim_list_index] == False):
            VSDout = [0]
        else:
            print run_data['interpolated_spike_voltage'][stim_list_index]
            print run_data['interpolated_spike_threshold'][stim_list_index]
            VSDout = square_voltage_dist(run_data['interpolated_spike_voltage'][stim_list_index], 
                                         run_data['interpolated_spike_threshold'][stim_list_index], 
                                         experiment)
        VSD_list.append(VSDout) 


#    print 'VSD_list', VSD_list
    concatenateVSDList=concatenate(VSD_list)
    logging.info('VSD: %f' % mean(concatenateVSDList))

#    if fileName!=None:
#        print 'file name ins vsd is', fileName
#        with open(os.path.splitext(fileName)[0]+'VSD.txt', 'a+') as f:
#            savetxt(f, mean(concatenateVSDList))

#--------Put this in if you want to do diagnostic plots but you can only be doing one sweep
#    diagnostic_plot_one_sweep(voltage_list, threshold_list, AScurrentMatrix_list, modelGridSpikeTime_list, 
#                           modelSpikeInterpolatedTime_list, gridISIFromLastTargSpike_list, interpolatedISIFromLastTargSpike_list, 
#                           voltageOfModelAtGridBioSpike_list, threshOfModelAtGridBioSpike_list, voltageOfModelAtInterpolatedBioSpike_list, 
#                           thresholdOfModelAtInterpolatedBioSpike_list, 
#                           experiment.grid_spike_time_target_list, experiment.neuron, experiment.stim_list, VSD_list, mean(concatenateVSDList))
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
    
    print 'VSD list', concatenateVSDList     
    return mean(concatenateVSDList) 

def square_voltage_dist(voltageOfModelAtBioSpike_array, threshOfModelAtBioSpike_array, experiment):
    '''
    inputs
        voltageOfModelAtBioSpike_array and threshOfModelAtBioSpike_array: arrays of thresholds and voltage of model at every target (bio) spike
        experiment: a neuron experiment wrapping the neuron class.
    outputs
        out: sqared difference between voltage and threshold of model neuron at each target (bio) spike'''
       
    if len(voltageOfModelAtBioSpike_array)!=len(threshOfModelAtBioSpike_array): 
        raise Exception('The two time vectors for the VSD are not the same length')
    
    out=zeros(len(voltageOfModelAtBioSpike_array)) #initiate vector
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
    BP_time=arange(0,len(BP_v_list[0]))*the_neuron.dt
    plt.figure(figsize=(15,11))
    plt.subplot(4,1,1)
    plt.plot(BP_time,stim_list[0])
    plt.title('current injection')
    plt.subplot(4,1,2)
    plt.plot(BP_time,BP_v_list[0], label='model voltage',  lw=2)
    plt.plot(BP_time, BP_thresh_list[0], label='model threshold',  lw=2)
    plt.plot(grid_spike_time_target_list[0], ones(len(BP_modelGridSpikeTime_list[0]))*the_neuron.th_inf, label='thr_inf', lw=2)

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

