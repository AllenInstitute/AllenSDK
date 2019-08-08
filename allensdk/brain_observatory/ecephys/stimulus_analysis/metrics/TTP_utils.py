# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:26:20 2018
@author: Xiaoxuan Jia
"""

# utils for interspike interval analysis
import numpy as np
from scipy import stats

def onset_function(psth, time, threshold=0.33, delay=0.03):
    tmp = time[np.where(psth>=max(psth)*threshold)[0]]
    if len(np.where(tmp>delay)[0])>0:
        return tmp[np.where(tmp>delay)[0][0]]
    else:
        return np.nan

def get_onset_threshold(unit_psth,time,threshold=0.33):
    # unit_psth: neuron*time
    onset=[]
    for i in range(unit_psth.shape[0]):
        psth=unit_psth[i,:]
        onset.append(onset_function(psth, time, threshold=threshold))
    onset=np.array(onset)    
    return onset

def get_TTP(psth, time):
    """psth is unit by time; units in seconds"""
    # remove first 30ms to get rid of influence from previous stimuli
    select_idx=np.where((time>=0.03) & (time<=0.2))[0]
    new_time=time[select_idx]
    TTP = np.zeros(np.shape(psth)[0])
    for i in range(np.shape(psth)[0]):
        TTP[i] = new_time[np.argmax(psth[i,select_idx])]
    return TTP

def get_onset(unit_psth, PSTH_bintime, p_value=0.01):
    """response onset is defined by comparing to baseline 40 ms after stimulus onset,
    the response time point is significantly (larger than p_value) above baseline."""
    T_onset = np.zeros(np.shape(unit_psth)[0])
    for u in range(np.shape(unit_psth)[0]):
        FR = np.nanmean(unit_psth[u, :,:].sum(1)/(unit_psth.shape[-1]*PSTH_bintime)*1000.)
        # reshape to combine orientations and repeats
        # tmp is trial*time for each unit
        if len(np.shape(unit_psth))==4:
            tmp = unit_psth[u,:,:,:].reshape(np.shape(unit_psth)[1]*np.shape(unit_psth)[2], np.shape(unit_psth)[3])
        elif len(np.shape(unit_psth))==3:
            tmp = unit_psth[u,:,:]

        baseline = np.nanmean(tmp[:,:(40/PSTH_bintime)], axis=1)

        P=np.zeros(np.shape(tmp)[1])

        # p value for each time point compared to baseline distributions across trials [0,60]ms
        for i in range(np.shape(tmp)[1]):
            t,p = stats.ttest_ind(baseline, tmp[:,i])
            P[i]=p

        significant_idx = np.where((P<p_value) & (P>0))[0]
        consecutive = np.diff(significant_idx)
        significant_idx[np.where(consecutive==1)[0]]
        for tt in range(1,150/PSTH_bintime):  
            if len(np.where(consecutive==1)[0])>0:
                T_onset[u]=significant_idx[np.where(consecutive==tt)[0]][0]*PSTH_bintime
                break
    # return units in second
    return T_onset/1000.