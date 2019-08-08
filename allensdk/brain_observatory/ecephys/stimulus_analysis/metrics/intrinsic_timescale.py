# -*- coding: utf-8 -*-
"""
Created on Wed Aug 8 12:05:39 2019
@author: Xiaoxuan Jia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from scipy.stats import pearsonr
import psth_utils as pu


def get_psth(binarized, PSTH_bintime, value='count'):
    """Calculate PSTH averaged across trials based on binarized spike trains.
    binarized: units*condition*repeats*time
    PSTH_bintime: dividable by binarized.shape[-1]
    """
    unit_psth = binarized.reshape(binarized.shape[:-1] + (int(binarized.shape[-1]/PSTH_bintime), PSTH_bintime))
    if value=='mean':
        unit_psth = unit_psth.mean(-1)
    if value=='count':
        unit_psth = unit_psth.sum(-1)
    time = np.array(range(int(unit_binarized.shape[-1]/PSTH_bintime)))*PSTH_bintime/1000.
    return unit_psth, time

def cal_rsc(spikes, PSTH_bintime=25):
    # spikes is binarized activity of one probe
    # 25 ms bin for spike count
    unit_psth, time = get_psth(spikes, PSTH_bintime, value='count')
    unit_psth = unit_psth[:,:,:,:10]
    print(unit_psth.shape)

    # for a given stimulus condition, compute spike count correlation across trials with different time bins
    # for each neuron
    amo = np.zeros((unit_psth.shape[1],  unit_psth.shape[0], unit_psth.shape[3], unit_psth.shape[3]))*np.nan
    amo_p = np.zeros((unit_psth.shape[1], unit_psth.shape[0], unit_psth.shape[3], unit_psth.shape[3]))*np.nan
    amo_sig = np.zeros((unit_psth.shape[1], unit_psth.shape[0], unit_psth.shape[3], unit_psth.shape[3]))*np.nan

    for lum in range(unit_psth.shape[1]):
        # luminance of flash
        for idx in range(unit_psth.shape[0]):
            # units
            tmp=unit_psth[idx, lum, :, :]
            for i in np.arange(tmp.shape[1]-1):
                # time bins
                for j in np.arange(i+1, tmp.shape[1]):
                    # remove zero spike count bins across trials
                    good_trials = np.intersect1d(np.where(tmp[:,i]>0)[0], np.where(tmp[:,j]>0)[0])
                    r, p = pearsonr(tmp[good_trials,i], tmp[good_trials,j])
                    amo[lum, idx, i,j]=r
                    amo_p[lum, idx,i,j]=p
                    if p<0.05:
                        amo_sig[lum, idx,i,j]=1          
    rsc_time_matrix = amo
    rsc_time_matrix = np.nanmean(rsc_time_matrix, axis=0)
    return rsc_time_matrix

def fit_exp(rsc_time_matrix):
    
    intr = abs(rsc_time_matrix)
    tmp = np.nanmean(intr, axis=0)
    n=intr.shape[0]
    
    t = np.arange(len(tmp))[1:]
    y=gaussian_filter(np.nanmean(tmp, axis=0)[1:],0.8)
    
    p, amo = curve_fit(lambda t,a,b,c: a*np.exp(-1/b*t)+c,  t,  y,  p0=(-4, 2, 1), maxfev = 1000000000)

    a=p[0]
    b=p[1] # this is the intrinsic timescale
    c=p[2]
    y_std = np.nanstd(tmp, axis=0)[1:]/np.sqrt(n)
    return t, y, y_std, a, b, c


def plot(t, y, y_std, a, b, c):
    xspacing = 0.1
    xt = np.arange(min(t), max(t),xspacing)
    fit_y = a*np.exp(-1/b*xt)+c
    
    plt.plot(xt, fit_y, 'k-',)
    plt.errorbar(t, y, y_std)
    plt.show()
