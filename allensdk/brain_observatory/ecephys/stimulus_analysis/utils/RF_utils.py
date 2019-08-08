# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:30:17 2018
@author: Xiaoxuan Jia
"""

# utils for interspike interval analysis
import numpy as np
from scipy import stats
import platform
if int(platform.python_version()[0])>2:
    import _pickle as pk
else:
    import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt


def get_RF_map(FR, stim_table):
    """Get mean response map for all neurons."""
    if 'Pos_x' in list(stim_table.keys()):
        x_locs = np.unique(stim_table['Pos_x'])
        y_locs = np.unique(stim_table['Pos_y'])
    elif 'pos_x' in list(stim_table.keys()):
        x_locs = np.unique(stim_table['pos_x'])
        y_locs = np.unique(stim_table['pos_y'])
    elif "b'pos_x'" in list(stim_table.keys()):
        x_locs = np.unique(stim_table["b'pos_x'"])
        y_locs = np.unique(stim_table["b'pos_y'"])

    # calculate response map
    # sequentially from [-40,40] degree
    rf=[]
    for n in range(np.shape(FR)[0]):
        rf_map=np.zeros([len(x_locs),len(y_locs)])
        count=0
        for i in range(np.shape(FR)[1]):
            resp = FR[n,i]
            if 'Pos_x' in list(stim_table.keys()):
                x_pos = np.where(x_locs==stim_table['Pos_x'].values[i])[0][0]
                y_pos = np.where(y_locs==stim_table['Pos_y'].values[i])[0][0]
            elif 'pos_x' in list(stim_table.keys()):
                x_pos = np.where(x_locs==stim_table['pos_x'].values[i])[0][0]
                y_pos = np.where(y_locs==stim_table['pos_y'].values[i])[0][0]
            elif "b'pos_x'" in list(stim_table.keys()):
                x_pos = np.where(x_locs==stim_table["b'pos_x'"].values[i])[0][0]
                y_pos = np.where(y_locs==stim_table["b'pos_y'"].values[i])[0][0]
            rf_map[x_pos,y_pos]+=resp
            count+=1
        rf.append(rf_map/count)
    rf=np.array(rf)
    return rf

def get_RF_map_trials(FR, stim_table):
    """Get mean response map for all neurons."""
    if 'Pos_x' in list(stim_table.keys()):
        x_locs = np.unique(stim_table['Pos_x'])
        y_locs = np.unique(stim_table['Pos_y'])
    elif 'pos_x' in list(stim_table.keys()):
        x_locs = np.unique(stim_table['pos_x'])
        y_locs = np.unique(stim_table['pos_y'])
    elif "b'pos_x'" in list(stim_table.keys()):
        x_locs = np.unique(stim_table["b'pos_x'"])
        y_locs = np.unique(stim_table["b'pos_y'"])
    
    n_neurons = FR.shape[0]
    n_trials = FR.shape[1]/(x_locs.shape[0]*y_locs.shape[0])

    # calculate response map
    # sequentially from [-40,40] degree
    rf=np.zeros((n_neurons, x_locs.shape[0], y_locs.shape[0], n_trials))
    for n in range(n_neurons):
        rf_map_trial=np.zeros([len(x_locs),len(y_locs), n_trials])
        for idx_x, x in enumerate(x_locs):
            for idx_y, y in enumerate(y_locs):
                if 'Pos_x' in list(stim_table.keys()):
                    rf_map_trial[idx_x, idx_y, :] = FR[n, np.where((stim_table['Pos_x'].values==x) & 
                                                               (stim_table['Pos_y'].values==y))[0]]
                elif 'pos_x' in list(stim_table.keys()):
                    rf_map_trial[idx_x, idx_y, :] = FR[n, np.where((stim_table['pos_x'].values==x) & 
                                                               (stim_table['pos_y'].values==y))[0]]
                elif "b'pos_x'" in list(stim_table.keys()):
                    rf_map_trial[idx_x, idx_y, :] = FR[n, np.where((stim_table["b'pos_x'"].values==x) & 
                                                               (stim_table["b'pos_y'"].values==y))[0]]
        rf[n,:,:,:]=rf_map_trial
    return rf