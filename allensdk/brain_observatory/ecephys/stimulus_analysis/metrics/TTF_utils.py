# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:36:39 2019
@author: Xiaoxuan Jia
"""
#------------# time to first spike

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

def find_nearest(array, value, side="right"):
    idx = np.searchsorted(array, value, side=side)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def find_first_spike(train, start_time=30, end_time=200):
    return find_nearest(np.where(train[start_time:end_time]==1)[0]+start_time, start_time, side="left")

def compute_first_spike(spikes, start_time=30, end_time=200):
    """spikes: neuron*trial*time"""
    first_spike=np.zeros((spikes.shape[0], spikes.shape[1]))*np.NaN
    for n in range(spikes.shape[0]):
        for t in range(spikes.shape[1]):
            train = spikes[n,t,:]
            if sum(np.where(train[start_time:end_time]==1)[0]+start_time)>0:
                first_spike[n,t]=find_first_spike(train, start_time=start_time, end_time=end_time)
    return first_spike

def compute_mean_first_spike(first_spike):
    # first_spike: n*trial
    mu=[]
    for n in np.arange(first_spike.shape[0]):
        mu.append(np.median(first_spike[n, np.where(first_spike[n,:]>0)]))
    mu=np.array(mu)
    return mu

