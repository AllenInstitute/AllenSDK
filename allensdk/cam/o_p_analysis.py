import scipy.stats as st
import numpy as np
import pandas as pd
import time
import os

from allensdk.cam.findlevel import findlevel
from cam_exceptions import CamAnalysisException

class OPAnalysis(object):
    def __init__(self, cam_analysis,
                 **kwargs):
        self.cam_analysis = cam_analysis
        self.save_dir = os.path.dirname(self.cam_analysis.save_path)
        
        self.timestamps, self.celltraces = self.cam_analysis.nwb.get_fluorescence_traces()
        self.numbercells = len(self.celltraces)                         #number of cells in dataset       
        self.acquisition_rate = 1/(self.timestamps[1]-self.timestamps[0])
        self.dxcm, self.dxtime = self.cam_analysis.nwb.get_running_speed()        
#        self.celltraces_dff = self.get_global_dff(percentiletosubtract=8)
#        self.binned_dx_sp, self.binned_cells_sp, self.binned_dx_vis, self.binned_cells_vis = self.get_speed_tuning(binsize=400)

    def get_response(self):
        raise CamAnalysisException("get_response not implemented")

    def get_peak(self):
        raise CamAnalysisException("get_peak not implemented")

    def get_global_dff(self, percentiletosubtract=8):
        '''does a global DF/F using a sliding window (+/- 15 s) baseline subtraction followed by Fo=peak of histogram'''
        '''replace when DF/F added to nwb file'''        
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
    
    def get_speed_tuning(self, binsize):
        print 'Calculating speed tuning, spontaneous vs visually driven'
        celltraces_trimmed = np.delete(self.celltraces_dff, range(len(self.dxcm), np.size(self.celltraces_dff,1)), axis=1) 
        #pull out spontaneous epoch(s)        
        spontaneous = self.cam_analysis.nwb.get_stimulus_table('spontaneous')

        peak_run = pd.DataFrame(index=range(self.numbercells), columns=('speed_max_sp','speed_min_sp','ptest_sp', 'mod_sp','speed_max_vis','speed_min_vis','ptest_vis', 'mod_vis'))
        
        dx_sp = self.dxcm[spontaneous.start.iloc[-1]:spontaneous.end.iloc[-1]]
        celltraces_sp = celltraces_trimmed[:,spontaneous.start.iloc[-1]:spontaneous.end.iloc[-1]]
        dx_vis = np.delete(self.dxcm, np.arange(spontaneous.start.iloc[-1],spontaneous.end.iloc[-1]))
        celltraces_vis = np.delete(celltraces_trimmed, np.arange(spontaneous.start.iloc[-1],spontaneous.end.iloc[-1]), axis=1)
        if len(spontaneous) > 1:
            dx_sp = np.append(dx_sp, self.dxcm[spontaneous.start.iloc[-2]:spontaneous.end.iloc[-2]], axis=0)
            celltraces_sp = np.append(celltraces_sp,celltraces_trimmed[:,spontaneous.start.iloc[-2]:spontaneous.end.iloc[-2]], axis=1)
            dx_vis = np.delete(dx_vis, np.arange(spontaneous.start.iloc[-2],spontaneous.end.iloc[-2]))
            celltraces_vis = np.delete(celltraces_vis, np.arange(spontaneous.start.iloc[-2],spontaneous.end.iloc[-2]), axis=1)
        celltraces_vis = celltraces_vis[:,~np.isnan(dx_vis)]
        dx_vis = dx_vis[~np.isnan(dx_vis)]  
        
        nbins = 1 + len(np.where(dx_sp>=1)[0])/binsize
        dx_sorted = dx_sp[np.argsort(dx_sp)]
        celltraces_sorted_sp = celltraces_sp[:, np.argsort(dx_sp)]
        binned_cells_sp = np.zeros((self.numbercells, nbins, 2))
        binned_dx_sp = np.zeros((nbins,2))
        for i in range(nbins):
            offset = findlevel(dx_sorted,1,'up')        
            if i==0:
                binned_dx_sp[i,0] = np.mean(dx_sorted[:offset])
                binned_dx_sp[i,1] = np.std(dx_sorted[:offset]) / np.sqrt(offset)            
                binned_cells_sp[:,i,0] = np.mean(celltraces_sorted_sp[:,:offset], axis=1)
                binned_cells_sp[:,i,1] = np.std(celltraces_sorted_sp[:,:offset], axis=1) / np.sqrt(offset)
            else:
                start = offset + (i-1)*binsize
                binned_dx_sp[i,0] = np.mean(dx_sorted[start:start+binsize])
                binned_dx_sp[i,1] = np.std(dx_sorted[start:start+binsize])/np.sqrt(binsize)
                binned_cells_sp[:,i,0] = np.mean(celltraces_sorted_sp[:, start:start+binsize], axis=1)
                binned_cells_sp[:,i,1] = np.std(celltraces_sorted_sp[:,start:start+binsize], axis=1)/np.sqrt(binsize)
        
        binned_cells_shuffled_sp = np.empty((self.numbercells, nbins, 2, 200))
        for shuf in range(200):
            celltraces_shuffled = celltraces_sp[:,np.random.permutation(np.size(celltraces_sp,1))]
            celltraces_shuffled_sorted = celltraces_shuffled[:, np.argsort(dx_sp)]
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
        dx_sorted = dx_vis[np.argsort(dx_vis)]
        celltraces_sorted_vis = celltraces_vis[:, np.argsort(dx_vis)]
        binned_cells_vis = np.zeros((self.numbercells, nbins, 2))
        binned_dx_vis = np.zeros((nbins,2))
        for i in range(nbins):
            offset = findlevel(dx_sorted,1,'up')        
            if i==0:
                binned_dx_vis[i,0] = np.mean(dx_sorted[:offset])
                binned_dx_vis[i,1] = np.std(dx_sorted[:offset]) / np.sqrt(offset)
                binned_cells_vis[:,i,0] = np.mean(celltraces_sorted_vis[:,:offset], axis=1)
                binned_cells_vis[:,i,1] = np.std(celltraces_sorted_vis[:,:offset], axis=1) / np.sqrt(offset)
            else:
                start = offset + (i-1)*binsize
                binned_dx_vis[i,0] = np.mean(dx_sorted[start:start+binsize])
                binned_dx_vis[i,1] = np.std(dx_sorted[start:start+binsize])/np.sqrt(binsize)
                binned_cells_vis[:,i,0] = np.mean(celltraces_sorted_vis[:, start:start+binsize], axis=1)
                binned_cells_vis[:,i,1] = np.std(celltraces_sorted_vis[:,start:start+binsize], axis=1)/np.sqrt(binsize)
        
        binned_cells_shuffled_vis = np.empty((self.numbercells, nbins, 2, 200))
        for shuf in range(200):
            celltraces_shuffled = celltraces_vis[:,np.random.permutation(np.size(celltraces_vis,1))]
            celltraces_shuffled_sorted = celltraces_shuffled[:, np.argsort(dx_vis)]
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
        
        return binned_dx_sp, binned_cells_sp, binned_dx_vis, binned_cells_vis, peak_run

    def get_sweep_response(self):
        '''calculates the response to each sweep and then for each stimulus condition'''
        def domean(x):
            return np.mean(x[self.interlength:self.interlength+self.sweeplength+self.extralength])#+1])
            
        def doPvalue(x):
            (_, p) = st.f_oneway(x[:self.interlength], x[self.interlength:self.interlength+self.sweeplength+self.extralength])
            return p
            
#        if self.h5path != None:
#            sweep_response = pd.read_hdf(self.h5path, 'sweep_response')
#            mean_sweep_response = pd.read_hdf(self.h5path, 'mean_sweep_response')
#        else:
        print 'Calculating responses for each sweep'        
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells+1)).astype(str))
        sweep_response.rename(columns={str(self.numbercells) : 'dx'}, inplace=True)
        for index, row in self.stim_table.iterrows():
            start = row['start'] - self.interlength
            end = row['start'] + self.sweeplength + self.interlength
            for nc in range(self.numbercells):
                temp = self.celltraces[nc,start:end]                                
                sweep_response[str(nc)][index] = 100*((temp/np.mean(temp[:self.interlength]))-1)
            sweep_response['dx'][index] = self.dxcm[start:end]   
        
        mean_sweep_response = sweep_response.applymap(domean)
        
        pval = sweep_response.applymap(doPvalue)
        return sweep_response, mean_sweep_response, pval            
        

