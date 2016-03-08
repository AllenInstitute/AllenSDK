# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:26:59 2015

@author: saskiad
"""

from pylab import *
import matplotlib
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
from Analysis.findlevel import *
import os
import time

class OPAnalysis(object):
    def __init__(self,*args,**kwargs):
        for k,v in kwargs.iteritems():
            setattr(self,k,v)
        self.LIMSID = LIMSID
        self.Cre = Cre
        self.HVA = HVA
        self.depth = depth
        self.exptpath = exptpath
        self.h5path = h5path
        self.datarate = datarate
        self.movie_name = movie_name
        for f in os.listdir(self.exptpath):
            if f.endswith('.nwb'):
                self.nwbpath = os.path.join(exptpath, f)
                print "NWB file:", f
        self.savepath = self.GetSavepath(self.exptpath)
        
        self.timestamps, self.celltraces = cn.get_Fluorescence_Traces(self.nwbpath)
        self.numbercells = len(self.celltraces)                         #number of cells in dataset       
        self.acquisition_rate = 1/(self.timestamps[1]-self.timestamps[0])
        self.dxcm, self.dxtime = cn.get_Running_Speed(self.nwbpath)        
#        self.celltraces_dff = self.getGlobalDFF(percentiletosubtract=8)
#        self.binned_dx_sp, self.binned_cells_sp, self.binned_dx_vis, self.binned_cells_vis = self.getSpeedTuning(binsize=400)
   
    def GetSavepath(self, exptpath):
        '''creates path used for saving figures and data'''
        savepath = os.path.join(exptpath, 'Data')
        if os.path.exists(savepath) == False:
            os.mkdir(savepath)
        else:
            print "Data folder already exists"
        return savepath

    def getGlobalDFF(self, percentiletosubtract=8):
        '''does a global DF/F using a sliding window (±15 s) baseline subtraction followed by Fo=peak of histogram'''
        '''replace when DF/F added to nwb file'''        
        if self.h5path!=None:
            try:
                celltraces_dff = op.loadh5(self.h5path, 'celltraces_dff')
                print "Loading global DF/F from Data.h5"
            except:
                pass
        else:  
            print "Calculating global DF/F ... this can take some time"
            startTime = time.time()
            celltraces_dff = np.zeros(self.celltraces.shape)
            for i in range(450):
                celltraces_dff[:,i] = self.celltraces[:,i] - np.percentile(self.celltraces[:,:(i+450)], percentiletosubtract, axis=1)
            for i in range(450, np.size(self.celltraces,1)-450):
                celltraces_dff[:,i] = self.celltraces[:,i] - np.percentile(self.celltraces[:,(i-450):(i+450)], percentiletosubtract, axis=1)
            for i in range(np.size(self.celltraces,1)-450, np.size(self.celltraces,1)):
                celltraces_dff[:,i] = self.celltraces[:,i] - np.percentile(self.celltraces[:,(i-450):], percentiletosubtract, axis=1)
    
            print "we're still here"
            for cn in range(self.numbercells):
                (val, edges) = np.histogram(celltraces_dff[cn,:], bins=200)
                celltraces_dff[cn,:] /= edges[np.argmax(val)+1]
                celltraces_dff[cn,:] -= 1
                celltraces_dff[cn,:] *= 100
            elapsedTime = time.time() - startTime
            print "Elapsed Time:", str(elapsedTime)
        return celltraces_dff
    
    def getSpeedTuning(self, binsize):
        print 'Calculating speed tuning, spontaneous vs visually driven'
        celltraces_trimmed = np.delete(self.celltraces_dff, range(len(self.dxcm), np.size(self.celltraces_dff,1)), axis=1) 
        #pull out spontaneous epoch(s)        
        spontaneous = cn.get_Stimulus_Table(self.nwbpath, 'spontaneous')

        peak_run = pd.DataFrame(index=range(self.numbercells), columns=('speed_max_sp','speed_min_sp','ptest_sp', 'mod_sp','speed_max_vis','speed_min_vis','ptest_vis', 'mod_vis'))
        peak_run['LIMS'] = self.LIMSID
        peak_run['Cre'] = self.Cre   
        peak_run['HVA'] = self.HVA
        peak_run['depth'] = self.depth        
        
        dx_sp = self.dxcm[spontaneous.Start.iloc[-1]:spontaneous.End.iloc[-1]]
        celltraces_sp = celltraces_trimmed[:,spontaneous.Start.iloc[-1]:spontaneous.End.iloc[-1]]
        dx_vis = np.delete(self.dxcm, np.arange(spontaneous.Start.iloc[-1],spontaneous.End.iloc[-1]))
        celltraces_vis = np.delete(celltraces_trimmed, np.arange(spontaneous.Start.iloc[-1],spontaneous.End.iloc[-1]), axis=1)
        if len(spontaneous) > 1:
            dx_sp = np.append(dx_sp, self.dxcm[spontaneous.Start.iloc[-2]:spontaneous.End.iloc[-2]], axis=0)
            celltraces_sp = np.append(celltraces_sp,celltraces_trimmed[:,spontaneous.Start.iloc[-2]:spontaneous.End.iloc[-2]], axis=1)
            dx_vis = np.delete(dx_vis, np.arange(spontaneous.Start.iloc[-2],spontaneous.End.iloc[-2]))
            celltraces_vis = np.delete(celltraces_vis, np.arange(spontaneous.Start.iloc[-2],spontaneous.End.iloc[-2]), axis=1)
        celltraces_vis = celltraces_vis[:,~np.isnan(dx_vis)]
        dx_vis = dx_vis[~np.isnan(dx_vis)]  
        
        nbins = 1 + len(np.where(dx_sp>=1)[0])/binsize
        dx_sorted = dx_sp[argsort(dx_sp)]
        celltraces_sorted_sp = celltraces_sp[:, argsort(dx_sp)]
        binned_cells_sp = np.zeros((self.numbercells, nbins, 2))
        binned_dx_sp = np.zeros((nbins,2))
        for i in range(nbins):
            offset = findlevel(dx_sorted,1,'up')        
            if i==0:
                binned_dx_sp[i,0] = np.mean(dx_sorted[:offset])
                binned_dx_sp[i,1] = np.std(dx_sorted[:offset])            
                binned_cells_sp[:,i,0] = np.mean(celltraces_sorted_sp[:,:offset], axis=1)
                binned_cells_sp[:,i,1] = np.std(celltraces_sorted_sp[:,:offset], axis=1)
            else:
                start = offset + (i-1)*binsize
                binned_dx_sp[i,0] = np.mean(dx_sorted[start:start+binsize])
                binned_dx_sp[i,1] = np.std(dx_sorted[start:start+binsize])/sqrt(binsize)
                binned_cells_sp[:,i,0] = np.mean(celltraces_sorted_sp[:, start:start+binsize], axis=1)
                binned_cells_sp[:,i,1] = np.std(celltraces_sorted_sp[:,start:start+binsize], axis=1)/sqrt(binsize)
        
        binned_cells_shuffled_sp = np.empty((self.numbercells, nbins, 2, 200))
        for shuf in range(200):
            celltraces_shuffled = celltraces_sp[:,np.random.permutation(size(celltraces_sp,1))]
            celltraces_shuffled_sorted = celltraces_shuffled[:, argsort(dx_sp)]
            for i in range(nbins):
                offset = findlevel(dx_sorted,1,'up')        
                if i==0:          
                    binned_cells_shuffled_sp[:,i,0,shuf] = np.mean(celltraces_shuffled_sorted[:,:offset], axis=1)
                    binned_cells_shuffled_sp[:,i,1,shuf] = np.std(celltraces_shuffled_sorted[:,:offset], axis=1)
                else:
                    start = offset + (i-1)*binsize
                    binned_cells_shuffled_sp[:,i,0,shuf] = np.mean(celltraces_shuffled_sorted[:, start:start+binsize], axis=1)
                    binned_cells_shuffled_sp[:,i,1,shuf] = np.std(celltraces_shuffled_sorted[:,start:start+binsize], axis=1)
                
        
        nbins = 1 + len(np.where(dx_vis>=1)[0])/binsize
        dx_sorted = dx_vis[argsort(dx_vis)]
        celltraces_sorted_vis = celltraces_vis[:, argsort(dx_vis)]
        binned_cells_vis = np.zeros((self.numbercells, nbins, 2))
        binned_dx_vis = np.zeros((nbins,2))
        for i in range(nbins):
            offset = findlevel(dx_sorted,1,'up')        
            if i==0:
                binned_dx_vis[i,0] = np.mean(dx_sorted[:offset])
                binned_dx_vis[i,1] = np.std(dx_sorted[:offset])            
                binned_cells_vis[:,i,0] = np.mean(celltraces_sorted_vis[:,:offset], axis=1)
                binned_cells_vis[:,i,1] = np.std(celltraces_sorted_vis[:,:offset], axis=1)
            else:
                start = offset + (i-1)*binsize
                binned_dx_vis[i,0] = np.mean(dx_sorted[start:start+binsize])
                binned_dx_vis[i,1] = np.std(dx_sorted[start:start+binsize])/sqrt(binsize)
                binned_cells_vis[:,i,0] = np.mean(celltraces_sorted_vis[:, start:start+binsize], axis=1)
                binned_cells_vis[:,i,1] = np.std(celltraces_sorted_vis[:,start:start+binsize], axis=1)/sqrt(binsize)
        
        binned_cells_shuffled_vis = np.empty((self.numbercells, nbins, 2, 200))
        for shuf in range(200):
            celltraces_shuffled = celltraces_vis[:,np.random.permutation(size(celltraces_vis,1))]
            celltraces_shuffled_sorted = celltraces_shuffled[:, argsort(dx_vis)]
            for i in range(nbins):
                offset = findlevel(dx_sorted,1,'up')        
                if i==0:          
                    binned_cells_shuffled_vis[:,i,0,shuf] = np.mean(celltraces_shuffled_sorted[:,:offset], axis=1)
                    binned_cells_shuffled_vis[:,i,1,shuf] = np.std(celltraces_shuffled_sorted[:,:offset], axis=1)
                else:
                    start = offset + (i-1)*binsize
                    binned_cells_shuffled_vis[:,i,0,shuf] = np.mean(celltraces_shuffled_sorted[:, start:start+binsize], axis=1)
                    binned_cells_shuffled_vis[:,i,1,shuf] = np.std(celltraces_shuffled_sorted[:,start:start+binsize], axis=1)
         
        shuffled_variance_sp = binned_cells_shuffled_sp[:,:,0,:].std(axis=1)**2
        variance_threshold_sp = np.percentile(shuffled_variance_sp, 99.9, axis=1)
        response_variance_sp = binned_cells_sp[:,:,0].std(axis=1)**2

                 
        shuffled_variance_vis = binned_cells_shuffled_vis[:,:,0,:].std(axis=1)**2
        variance_threshold_vis = np.percentile(shuffled_variance_vis, 99.9, axis=1)
        response_variance_vis = binned_cells_vis[:,:,0].std(axis=1)**2
#        for nc in range(self.numbercells):
#            if response_variance_vis[nc]>variance_threshold_vis[nc]:
#                peak.mod_vis[nc] = True
#            if response_variance_vis[nc]<=variance_threshold_vis[nc]:
#                peak.mod_vis[nc] = False
#            if response_variance_sp[nc]>variance_threshold_sp[nc]:
#                peak.mod_sp[nc] = True
#            if repsonse_variance_sp[nc]<=variance_threshold_sp[nc]:
#                peak.mod_sp[nc] = False
        
#        if (100*float(len(np.where(self.dxcm>2)[0]))/len(self.dxcm))>2.:
#             run_start = findlevels(self.dxcm, threshold=2, window=10, direction='up')
#             cell_run = np.empty((len(run_start),self.numbercells,60))
#             del_pts = []
#             for i,v in enumerate(run_start):
#                 if v>30:
#                     cell_run[i,:,:] = self.celltraces_dff[:,v-30:v+30]
#                 else:
#                     del_pts.append(i)
#             cell_run = np.delete(cell_run,del_pts,axis=0)
         
        for nc in range(self.numbercells):
            if response_variance_vis[nc]>variance_threshold_vis[nc]:
                peak_run.mod_vis[nc] = True
            if response_variance_vis[nc]<=variance_threshold_vis[nc]:
                peak_run.mod_vis[nc] = False
            if response_variance_sp[nc]>variance_threshold_sp[nc]:
                peak_run.mod_sp[nc] = True
            if response_variance_sp[nc]<=variance_threshold_sp[nc]:
                peak_run.mod_sp[nc] = False
            temp = binned_cells_sp[nc,:,0]
            start_max = temp.argmax()
            peak_run.speed_max_sp[nc] = binned_dx_sp[start_max,0]
            start_min = temp.argmin()
            peak_run.speed_min_sp[nc] = binned_dx_sp[start_min,0]
            if peak_run.speed_max_sp[nc]>peak_run.speed_min_sp[nc]:
                test_values = celltraces_sorted_sp[nc,start_max*binsize:(start_max+1)*binsize]
                other_values = np.delete(celltraces_sorted_sp[nc,:], range(start_max*binsize, (start_max+1)*binsize))
                (_ ,peak_run.ptest_sp[nc]) = st.ks_2samp(test_values, other_values)
            else:
                test_values = celltraces_sorted_sp[nc,start_min*binsize:(start_min+1)*binsize]
                other_values = np.delete(celltraces_sorted_sp[nc,:], range(start_min*binsize, (start_min+1)*binsize))
                (_ ,peak_run.ptest_sp[nc]) = st.ks_2samp(test_values, other_values)
            temp = binned_cells_vis[nc,:,0]
            start_max = temp.argmax()
            peak_run.speed_max_vis[nc] = binned_dx_vis[start_max,0]
            start_min = temp.argmin()
            peak_run.speed_min_vis[nc] = binned_dx_vis[start_min,0]
            if peak_run.speed_max_vis[nc]>peak_run.speed_min_vis[nc]:
                test_values = celltraces_sorted_vis[nc,start_max*binsize:(start_max+1)*binsize]
                other_values = np.delete(celltraces_sorted_vis[nc,:], range(start_max*binsize, (start_max+1)*binsize))
            else:  
                test_values = celltraces_sorted_vis[nc,start_min*binsize:(start_min+1)*binsize]
                other_values = np.delete(celltraces_sorted_vis[nc,:], range(start_min*binsize, (start_min+1)*binsize))
            (_ ,peak_run.ptest_vis[nc]) = st.ks_2samp(test_values, other_values)
#             (_, peak_run.rta_ptest[nc]) = st.ks_2samp(cell_run[:,nc,:30].flatten(), cell_run[:,nc,30:].flatten())
#             (_, peak_run.rta_modulation[nc]) = cell_run[:,nc,30:].flatten().mean() / cell_run[:,nc,:30].flatten().mean()
        peak_run.to_csv(os.path.join(self.savepath, 'peak_Speed.csv'))             
        return binned_dx_sp, binned_cells_sp, binned_dx_vis, binned_cells_vis, peak_run

    def getSweepResponse(self):
        '''calculates the response to each sweep and then for each stimulus condition'''
        def domean(x):
            return np.mean(x[self.interlength:self.interlength+self.sweeplength+self.extralength])#+1])
            
        def doPvalue(x):
            (f, p) = st.f_oneway(x[:self.interlength], x[self.interlength:self.interlength+self.sweeplength+self.extralength])
            return p
            
#        if self.h5path != None:
#            sweep_response = pd.read_hdf(self.h5path, 'sweep_response')
#            mean_sweep_response = pd.read_hdf(self.h5path, 'mean_sweep_response')
#        else:
        print 'Calculating responses for each sweep'        
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells+1)).astype(str))
        sweep_response.rename(columns={str(self.numbercells) : 'dx'}, inplace=True)
        for index, row in self.stim_table.iterrows():
            start = row['Start'] - self.interlength
            end = row['Start'] + self.sweeplength + self.interlength
            for nc in range(self.numbercells):
                temp = self.celltraces[nc,start:end]                                
                sweep_response[str(nc)][index] = 100*((temp/np.mean(temp[:self.interlength]))-1)
            sweep_response['dx'][index] = self.dxcm[start:end]   
        
        mean_sweep_response = sweep_response.applymap(domean)
        
        pval = sweep_response.applymap(doPvalue)
        return sweep_response, mean_sweep_response, pval            
        
    def testPtest(self):
        '''running new ptest'''
        test = pd.DataFrame(index=self.sweeptable.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for nc in range(self.numbercells):        
            for index, row in self.sweeptable.iterrows():
                ori=row['Ori']
                tf=row['TF']
                test[str(nc)][index] = self.mean_sweep_response[(self.sync_table['TF']==tf)&(self.sync_table['Ori']==ori)][str(nc)]
        ptest = []
        for nc in range(self.numbercells):
            groups = []
            for index,row in test.iterrows():
                groups.append(test[str(nc)][index])
                (f,p) = st.f_oneway(*groups)
            ptest.append(p)
        ptest = np.array(ptest)
        cells = list(np.where(ptest<0.01)[0])
        print "# cells: " + str(len(ptest))
        print "# significant cells: " + str(len(cells))
        return ptest, cells
    
    def Ptest(self):
        ptest = np.empty((self.numbercells))
        for nc in range(self.numbercells):
            groups = []
            for ori in self.orivals:
                for tf in self.tfvals:
                    groups.append(self.mean_sweep_response[(self.stim_table.Temporal_frequency==tf)&(self.stim_table.Orientation==ori)][str(nc)])
            f,p = st.f_oneway(*groups)
            ptest[nc] = p
        return ptest

class DriftingGrating(OPAnalysis):    
    def __init__(self, *args, **kwargs):
        super(DriftingGrating, self).__init__(*args, **kwargs)                   
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
        if self.h5path != None:
            response = op.loadh5(self.h5path, 'response')
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
        peak['LIMS'] = self.LIMSID
        peak['Cre'] = self.Cre   
        peak['HVA'] = self.HVA
        peak['depth'] = self.depth
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

class StaticGrating(OPAnalysis):
    def __init__(self, *args, **kwargs):
        super(StaticGrating, self).__init__(*args, **kwargs)
        stimulus_table = cn.get_Stimulus_Table(self.nwbpath, 'static_gratings')
        self.stim_table = stimulus_table.fillna(value=0.)     
        self.sweeplength = self.stim_table['End'].iloc[1] - self.stim_table['Start'].iloc[1]
        self.interlength = 4 * self.sweeplength
        self.extralength = self.sweeplength
        self.orivals = np.unique(self.stim_table.Orientation.dropna())
        self.sfvals = np.unique(self.stim_table.Spatial_frequency.dropna())
        self.phasevals = np.unique(self.stim_table.Phase.dropna())
        self.number_ori = len(self.orivals)
        self.number_sf = len(self.sfvals)
        self.number_phase = len(self.phasevals)            
        self.sweep_response, self.mean_sweep_response, self.pval = self.getSweepResponse()
        self.response = self.getResponse()
        self.peak = self.getPeak()
#        self.binned_dx_sp, self.binned_cells_sp, self.binned_dx_vis, self.binned_cells_vis = self.getSpeedTuning(binsize=200)
        
#    
    def getResponse(self):
#        if self.h5path != None:
#            response = op.loadh5(self.h5path, 'response')
        print "Calculating mean responses"
        response = np.empty((self.number_ori, self.number_sf, self.number_phase, self.numbercells+1, 3))
#        blank = np.empty((self.numbercells, 3))        
        def ptest(x):
            return len(np.where(x<(0.05/(self.number_ori*(self.number_sf-1))))[0])
            
        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]
            for sf in self.sfvals:
                sf_pt = np.where(self.sfvals == sf)[0][0]
                for phase in self.phasevals:
                    phase_pt = np.where(self.phasevals == phase)[0][0]
                    subset_response = self.mean_sweep_response[(self.stim_table.Spatial_frequency==sf)&(self.stim_table.Orientation==ori)&(self.stim_table.Phase==phase)]
                    subset_pval = self.pval[(self.stim_table.Spatial_frequency==sf)&(self.stim_table.Orientation==ori)&(self.stim_table.Phase==phase)]
                    response[ori_pt, sf_pt, phase_pt, :, 0] = subset_response.mean(axis=0)
                    response[ori_pt, sf_pt, phase_pt, :, 1] = subset_response.std(axis=0)/sqrt(len(subset_response))
                    response[ori_pt, sf_pt, phase_pt, :, 2] = subset_pval.apply(ptest, axis=0)
#        subset_response = self.mean_sweep_response[self.stim_table.Orientation.isnull()]
#        subset_pval = self.pval[self.stim_table.Orientation.isnull()]
#        blank[:,0] = subset_response.mean(axis=0)
#        blank[:,1] = subset_response.std(axis=0)/sqrt(len(subset_response))
#        blank[:,2] = subset_pval.apply(ptest, axis=0)
        return response#, blank
    
    def getPeak(self):    
        '''finds the peak response for each cell'''
        print 'Calculating peak response properties'
        peak = pd.DataFrame(index=range(self.numbercells), columns=('Ori','SF', 'Phase', 'sg_response_variability','OSI','sg_peak_DFF','ptest'))
        peak['LIMS'] = self.LIMSID
        peak['Cre'] = self.Cre   
        peak['HVA'] = self.HVA
        peak['depth'] = self.depth
        for nc in range(self.numbercells):
            cell_peak = np.where(self.response[:,1:,:,nc,0] == np.nanmax(self.response[:,1:,:,nc,0]))
            pref_ori = cell_peak[0][0]
            pref_sf = cell_peak[1][0]+1
            pref_phase = cell_peak[2][0]
            peak.Ori[nc] = pref_ori
            peak.SF[nc] = pref_sf
            peak.Phase[nc] = pref_phase
            peak.sg_response_variability[nc] = self.response[pref_ori, pref_sf, pref_phase, nc, 2]/0.48  #TODO: check number of trials
            pref = self.response[pref_ori, pref_sf, pref_phase, nc, 0]
            orth = self.response[np.mod(pref_ori+3, 6), pref_sf, pref_phase, nc, 0]
            peak.OSI[nc] = (pref-orth)/(pref+orth)
            peak.sg_peak_DFF[nc] = pref
            groups = []
            for ori in self.orivals:
                for sf in self.sfvals:
                    for phase in self.phasevals:
                        groups.append(self.mean_sweep_response[(self.stim_table.Spatial_frequency==sf)&(self.stim_table.Orientation==ori)&(self.stim_table.Phase==phase)][str(nc)])
            f,p = st.f_oneway(*groups)
            peak.ptest[nc] = p
        peak.to_csv(os.path.join(self.savepath, 'Peak_static_grating.csv'))
            #TODO: add ptest, reliability, etc
#        peak['ptest'] = self.ptest
        return peak
    
#    def Ptest(self):
#        '''running new ptest'''
#        test = pd.DataFrame(index=self.sweeptable.index.values, columns=np.array(range(self.numbercells)).astype(str))
#        for nc in range(self.numbercells):        
#            for index, row in self.sweeptable.iterrows():
#                ori=row.Ori
#                sf=row.SF
#                phase = row.Phase
#                test[str(nc)][index] = self.mean_sweep_response[(self.stim_table.Spatial_frequency==sf)&(self.stim_table.Orientation==ori)&(self.stim_table.Phase==phase)][str(nc)]
#        ptest = []
#        for nc in range(self.numbercells):
#            groups = []
#            for index,row in test.iterrows():
#                groups.append(test[str(nc)][index])
#                (f,p) = st.f_oneway(*groups)
#            ptest.append(p)
#        ptest = np.array(ptest)
#        cells = list(np.where(ptest<0.01)[0])
#        print "# cells: " + str(len(ptest))
#        print "# significant cells: " + str(len(cells))
#        return ptest, cells


class NaturalImages(OPAnalysis):
    def __init__(self, *args, **kwargs):
        super(NaturalImages, self).__init__(*args, **kwargs)
        self.stim_table = cn.get_Stimulus_Table(self.nwbpath, 'natural_scenes')        
        self.number_images = len(np.unique(self.stim_table.Frame))
        self.sweeplength = self.stim_table.End.iloc[1] - self.stim_table.Start.iloc[1]
        self.interlength = 3 * self.sweeplength
        self.extralength = self.sweeplength
        self.sweep_response, self.mean_sweep_response, self.pval = self.getSweepResponse()
        self.response = self.getResponse()
        self.peak = self.getPeak()
        
    def getResponse(self):
        print "Calculating mean responses"
        response = np.empty((self.number_images, self.numbercells+1, 3))
        def ptest(x):
            return len(np.where(x<(0.05/(self.number_images-1)))[0])
            
        for ni in range(self.number_images):
            subset_response = self.mean_sweep_response[self.stim_table.Frame==(ni-1)]
            subset_pval = self.pval[self.stim_table.Frame==(ni-1)]            
            response[ni,:,0] = subset_response.mean(axis=0)
            response[ni,:,1] = subset_response.std(axis=0)/np.sqrt(len(subset_response))
            response[ni,:,2] = subset_pval.apply(ptest, axis=0)
        return response
    
    def getPeak(self):    
        '''gets metrics about peak response, etc.'''
        print 'Calculating peak response properties'
        peak = pd.DataFrame(index=range(self.numbercells), columns=('Image', 'response_variability','peak_DFF', 'ptest', 'p_run', 'run_modulation', 'time_to_peak','duration'))
        peak['LIMS'] = self.LIMSID
        peak['Cre'] = self.Cre   
        peak['HVA'] = self.HVA
        peak['depth'] = self.depth
        for nc in range(self.numbercells):
            nip = np.argmax(self.response[1:,nc,0])
            peak.Image[nc] = nip
            peak.response_variability[nc] = self.response[nip+1,nc,2]/0.50 #assume 50 trials
            peak.peak_DFF[nc] = self.response[nip+1,nc,0]
            subset = self.mean_sweep_response[self.stim_table.Frame==nip]
#            blank = self.mean_sweep_response[self.stim_table.Frame==-1]
            subset_stat = subset[subset.dx<2]
            subset_run = subset[subset.dx>=2]
            if (len(subset_run)>5) & ( len(subset_stat)>5):
                (f, peak.p_run[nc]) = st.ks_2samp(subset_run[str(nc)], subset_stat[str(nc)])
                peak.run_modulation[nc] = subset_run[str(nc)].mean()/subset_stat[str(nc)].mean()
            else:
                peak.p_run[nc] = np.NaN
                peak.run_modulation[nc] = np.NaN
            groups = []
            for im in range(self.number_images):
                subset = self.mean_sweep_response[self.stim_table.Frame==(im-1)]
                groups.append(subset[str(nc)].values)
            (f,peak.ptest[nc]) = st.f_oneway(*groups)
            test = self.sweep_response[self.stim_table.Frame==nip][str(nc)].mean()
            peak.time_to_peak[nc] = (np.argmax(test) - self.interlength)/self.acquisition_rate
            test2 = np.where(test<(test.max()/2))[0]          
            peak.duration[nc] = np.ediff1d(test2).max()/self.acquisition_rate
        peak.to_csv(os.path.join(self.savepath, 'peak_natural images.csv'))
        return peak

class MovieAnalysis(OPAnalysis):    
    def __init__(self, *args, **kwargs):
        super(MovieAnalysis, self).__init__(*args, **kwargs)                   
        stimulus_table = cn.get_Stimulus_Table(self.nwbpath, self.movie_name)   
        self.stim_table = stimulus_table[stimulus_table.Frame==0]
        self.celltraces_dff = self.getGlobalDFF(percentiletosubtract=8)
        if self.movie_name == 'natural_movie_one':
            self.binned_dx_sp, self.binned_cells_sp, self.binned_dx_vis, self.binned_cells_vis, self.peak_run = self.getSpeedTuning(binsize=800)
        self.sweeplength = self.stim_table.Start.iloc[1] - self.stim_table.Start.iloc[0]
        self.sweep_response = self.getSweepResponse()
        self.peak = self.getPeak()     
        
    def getSweepResponse(self):
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for index, row in self.stim_table.iterrows():
            start = row.Start
            end = start + self.sweeplength
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.celltraces_dff[nc,start:end]
        return sweep_response
    
    def getPeak(self):
        peak_movie = pd.DataFrame(index=range(self.numbercells), columns=('peak','response_variability'))
        for nc in range(self.numbercells):
            meanresponse = self.sweep_response[str(nc)].mean()
            movie_len= len(meanresponse)/30
            output = np.empty((movie_len,10))
            for tr in range(10):
                test = self.sweep_response[str(nc)].iloc[tr]
                for i in range(movie_len):
                    f,p = st.ks_2samp(test[i*30:(i+1)*30],test[(i+1)*30:(i+2)*30])
                    output[i,tr] = p    
            output = np.where(output<0.05, 1, 0)
            ptime = np.sum(output, axis=1)
            ptime*=10
            peak = np.argmax(meanresponse)
            if peak>30:
                peak_movie.response_variability[nc] = ptime[(peak-30)/30]
            else:
                peak_movie.response_variability[nc] = ptime[0]
            peak_movie.peak[nc] = peak
        peak_movie.to_csv(os.path.join(self.savepath, 'peak'+self.movie_name+'.csv'))
        return peak_movie

class LocallySN(OPAnalysis):    
    def __init__(self, *args, **kwargs):
        super(LocallySN, self).__init__(*args, **kwargs)        
        self.stim_table = cn.get_Stimulus_Table(self.nwbpath, 'locally_sparse_noise')
        self.LSN = cn.get_Stimulus_Template(self.nwbpath, 'locally_sparse_noise')
        self.sweeplength = self.stim_table['End'][1] - self.stim_table['Start'][1]
        self.interlength = self.sweeplength
        self.extralength = self.sweeplength
        self.sweep_response, self.mean_sweep_response, self.pval = self.getSweepResponse()
        self.receptive_field = self.getReceptiveField()
    
#    def getSweepResponse(self):
#        '''calculates the response to each sweep and then for each stimulus condition'''
#        def domean(x):
#            return np.mean(x[self.sweeplength:(3*self.sweeplength)+1])
#        
#        def doPvalue(x):
#            (f, p) = st.f_oneway(x[:self.sweeplength], x[self.sweeplength:(3*self.sweeplength)])
#            return p
#            
#        if self.h5path != None:
#            sweep_response = pd.read_hdf(self.h5path, 'sweep_response')
#            mean_sweep_response = pd.read_hdf(self.h5path, 'mean_sweep_response')
#        else:        
#            print 'Calculating responses for each sweep'        
#            sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells+1)).astype(str))
#            sweep_response.rename(columns={str(self.numbercells) : 'dx'}, inplace=True)
#            for nc in range(self.numbercells):
#                for index, row in self.stim_table.iterrows():
#                    start = row['Start'] - self.sweeplength
#                    end = row['Start'] + (2*self.sweeplength)
#    #                try:
#                    temp = self.celltraces[nc,start:end]                                
#                    sweep_response[str(nc)][index] = 100*((temp/np.mean(temp[:self.sweeplength]))-1)
#    #                except:
#    #                    sweep_response['dx'][index] = self.dxcm[start:end]
#            mean_sweep_response = sweep_response.applymap(domean)
##        pval = sweep_response.applymap(doPvalue)
#        pval = []
#        return sweep_response, mean_sweep_response, pval
#    
    def getReceptiveField(self):
        if self.h5path != None:
            receptive_field = op.loadh5(self.h5path, 'receptive_field')
        else:
            print "Calculating mean responses"
            receptive_field = np.empty((16, 28, self.numbercells+1, 2))
    #        def ptest(x):
    #            return len(np.where(x<(0.05/(8*5)))[0])
            for xp in range(16):
                for yp in range(28):
                    on_frame = np.where(self.LSN[:,xp,yp]==255)[0]
                    off_frame = np.where(self.LSN[:,xp,yp]==0)[0]
                    subset_on = self.mean_sweep_response[self.stim_table.Frame.isin(on_frame)]
                    subset_off = self.mean_sweep_response[self.stim_table.Frame.isin(off_frame)]
                    receptive_field[xp,yp,:,0] = subset_on.mean(axis=0)
                    receptive_field[xp,yp,:,1] = subset_off.mean(axis=0)
        return receptive_field  
if __name__=='__main__':
    #LIMSID = '501886692'
    LIMSID = '501836392'
    Cre = 'Cux2'
    HVA = 'AL'
    depth= 175
    cam_directory = '/local1/cam_datasets'               
    #exptpath = os.path.join(r's/saskiad/Documents/Data/ophysdev', LIMSID)
    exptpath = os.path.join(cam_directory, LIMSID)
    h5path = None
    datarate = 30    
    movie_name = 'natural_movie_one'
    dg = DriftingGrating(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)

#    sg = StaticGrating(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)    
    nm1 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)
#    ni = NaturalImages(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)
#    movie_name = 'natural_movie_two'    
#    nm2 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)
    movie_name = 'natural_movie_three'
    nm3 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)    
    #cp.plot_Movie_All(nm)
    #lsn = LocallySN(exptpath, h5path,datarate, LIMSID, Cre, HVA, depth)
    cp.plot_3SA(dg, nm1, nm3)
    cp.plot_Drifting_grating_Traces(dg)
    
#    print "Number of cells:", str(lsn.numbercells)
#    cp.plot_3SB(sg, nm1, ni)
#    cp.plot_NI_Traces(ni)
#    cp.plot_SG_Traces(sg)
    #cp.plot_3SC(lsn, nm1, nm2)
    #cp.plot_LSN_Traces(lsn)
    
#    lsn.plotAll()
#    dg.saveh5()

#    dg.plotTraces()   
#    dg.ExperimentSummary()
#    dg.plotRunning()    
    
    
#
##                
#filename = 'Data.h5'
#fullfilename = os.path.join(sg.savepath, filename)
#store = pd.HDFStore(fullfilename)
#store['stim_table_sg'] = sg.stim_table
#store['sweep_response_sg'] = sg.sweep_response
#store['mean_sweep_response_sg'] = sg.mean_sweep_response
#store['sweep_response_nm1'] = nm1.sweep_response
#store['stim_table_nm1'] = nm1.stim_table
#store['sweep_response_ni'] = ni.sweep_response
#store['stim_table_ni'] = ni.stim_table
#store['mean_sweep_response_ni'] = ni.mean_sweep_response
##store['sweeptable'] = dg.sweeptable
#store.close()
#f = h5py.File(fullfilename, 'r+')
##dset6 = f.create_dataset('receptive_field', data=lsn.receptive_field)
##dset = f.create_dataset('celltraces', data=sg.celltraces)
##dset2 = f.create_dataset('twop_frames', data=sg.twop_frames)
##dset3 = f.create_dataset('acquisition_rate', data=sg.acquisition_rate)
#dset4= f.create_dataset('celltraces_dff', data=nm1.celltraces_dff)
##dset5 = f.create_dataset('dxcm', data=sg.dxcm)  
#dset6 = f.create_dataset('response_sg', data=sg.response)
#dset5 = f.create_dataset('response_ni', data=ni.response)
#dset = f.create_dataset('binned_cells_sp', data=nm1.binned_cells_sp)
#dset1 = f.create_dataset('binned_cells_vis', data=nm1.binned_cells_vis)
#dset2 = f.create_dataset('binned_dx_sp', data=nm1.binned_dx_sp)
#dset3 = f.create_dataset('binned_dx_vis',data=nm1.binned_dx_vis)
###f.keys()
#f.close()
####            
