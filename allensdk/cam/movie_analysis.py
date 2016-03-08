import scipy.stats as st
import pandas as pd
import numpy as np
#import Analysis.OPTools_Nikon as op
import Analysis.CAM_NWB as cn
from allensdk.cam.o_p_analysis import OPAnalysis
import os

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
            #receptive_field = op.loadh5(self.h5path, 'receptive_field')
            raise(Exception('no loadh5'))
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
