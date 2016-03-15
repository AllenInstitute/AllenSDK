from pylab import *
import matplotlib
from o_p_analysis import OPAnalysis
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.ioff()
import scipy.stats as st
import numpy as np
import cPickle as pickle
import h5py
import pandas as pd
#import Analysis.OPTools_Nikon as op
import Analysis.CAM_NWB as cn
import CAM_plotting as cp
from allensdk.cam.Analysis.findlevel import *
import os

class DriftingGrating(OPAnalysis):    
    def __init__(self, cam_analysis, **kwargs):
        super(DriftingGrating, self).__init__(cam_analysis, **kwargs)                   
        stimulus_table = cn.get_Stimulus_Table(self.nwbpath, 'drifting_gratings')
        self.stim_table = stimulus_table.fillna(value=0.)     
        self.sweeplength = 60#self.sync_table['end'][1] - self.sync_table['start'][1]
        self.interlength = 30#self.sync_table['start'][2] - self.sync_table['end'][1]
        self.extralength = 0
        self.orivals = np.unique(self.stim_table.Orientation).astype(int)
        self.tfvals = np.unique(self.stim_table.Temporal_frequency).astype(int)
        self.number_ori = len(self.orivals)
        self.number_tf = len(self.tfvals)            
        self.sweep_response, self.mean_sweep_response, self.pval = self.getSweepResponse()
        self.response = self.getResponse()
        self.peak = self.getPeak()
    
    def getResponse(self):
        print "Calculating mean responses"
        response = np.empty((self.number_ori, self.number_tf, self.numbercells+1, 3))
        def ptest(x):
            return len(np.where(x<(0.05/(8*5)))[0])
            
        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]
            for tf in self.tfvals:
                tf_pt = np.where(self.tfvals == tf)[0][0]
                subset_response = self.mean_sweep_response[(self.stim_table.Temporal_frequency==tf)&(self.stim_table.Orientation==ori)]
                subset_pval = self.pval[(self.stim_table.Temporal_frequency==tf)&(self.stim_table.Orientation==ori)]
                response[ori_pt, tf_pt, :, 0] = subset_response.mean(axis=0)
                response[ori_pt, tf_pt, :, 1] = subset_response.std(axis=0)/sqrt(len(subset_response))
                response[ori_pt, tf_pt, :, 2] = subset_pval.apply(ptest, axis=0)
        return response
    
    def getPeak(self):
        '''finds the peak response for each cell'''
        print 'Calculating peak response properties'
        peak = pd.DataFrame(index=range(self.numbercells), columns=('Ori','TF','response_variability','OSI','DSI','peak_DFF', 'reliability', 'ptest'))
        peak['LIMS'] = self.cam_analysis.lims_id
        peak['Cre'] = self.Cre   
        peak['HVA'] = self.HVA
        peak['depth'] = self.cam_analysis.depth
        for nc in range(self.numbercells):
            cell_peak = np.where(self.response[:,1:,nc,0] == np.nanmax(self.response[:,1:,nc,0]))
            prefori = cell_peak[0][0]
            preftf = cell_peak[1][0]+1
            peak['Ori'][nc] = prefori
            peak['TF'][nc] = preftf
            peak['response_variability'][nc] = self.response[prefori, preftf, nc, 2]/0.15
            pref = self.response[prefori, preftf, nc, 0]            
            orth1 = self.response[np.mod(prefori+2, 8), preftf, nc, 0]
            orth2 = self.response[np.mod(prefori-2, 8), preftf, nc, 0]
            orth = (orth1+orth2)/2
            null = self.response[np.mod(prefori+4, 8), preftf, nc, 0]
            peak['OSI'][nc] = (pref-orth)/(pref+orth)
            peak['DSI'][nc] = (pref-null)/(pref+null)
            peak['peak_DFF'][nc] = pref
            
            pref_std = self.response[prefori, preftf, nc, 1]*sqrt(15)
            blank_mean = self.response[0, 0, nc, 0]
            blank_std = self.response[0, 0, nc, 1]*np.sqrt(len(self.stim_table[self.stim_table.Temporal_frequency==0]))
            peak['reliability'][nc] = (pref - blank_mean)/(pref_std + blank_std)
            
            groups = []
            for ori in self.orivals:
                for tf in self.tfvals:
                    groups.append(self.mean_sweep_response[(self.stim_table.Temporal_frequency==tf)&(self.stim_table.Orientation==ori)][str(nc)])
            f,p = st.f_oneway(*groups)
            peak.ptest[nc] = p
        peak.to_csv(os.path.join(self.savepath, 'peak_drifting_grating.csv'))
        return peak
    
    def getRunModulation(self, speed_threshold=10):
        print 'Calculating run modulation at peak'
#        run_modulation = np.empty((self.numbercells,4))
        run_modulation = pd.DataFrame(index=range(self.numbercells), columns=('stationary_mean','stationary_sem','run_mean','run_sem'))
        
        for nc in range(self.numbercells):
            ori = self.orivals[self.peak['Ori'][nc]]
            tf = self.tfvals[self.peak['TF'][nc]]
            subset_response = self.mean_sweep_response[(self.sync_table['TF']==tf)&(self.sync_table['Ori']==ori)]
            subset_run = subset_response[subset_response['dx'] >= speed_threshold]
            subset_stationary = subset_response[subset_response['dx'] < speed_threshold]
            run_modulation['stationary_mean'][nc] = subset_stationary[str(nc)].mean()
            run_modulation['stationary_sem'][nc] = subset_stationary[str(nc)].std()/sqrt(len(subset_stationary))
            run_modulation['run_mean'][nc] = subset_run[str(nc)].mean()
            run_modulation['run_sem'][nc] = subset_run[str(nc)].std()/sqrt(len(subset_run))
        
        return run_modulation
