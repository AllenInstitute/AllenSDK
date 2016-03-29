import scipy.stats as st
import pandas as pd
import numpy as np
from allensdk.cam.o_p_analysis import OPAnalysis

class MovieAnalysis(OPAnalysis):    
    def __init__(self, cam_analysis, movie_name, **kwargs):
        super(MovieAnalysis, self).__init__(cam_analysis, **kwargs)                   
        stimulus_table = self.cam_analysis.nwb.get_stimulus_table(movie_name)   
        self.stim_table = stimulus_table[stimulus_table.frame==0]
        self.celltraces_dff = self.getGlobalDFF(percentiletosubtract=8)
        if movie_name == 'natural_movie_one':
            self.binned_dx_sp, self.binned_cells_sp, self.binned_dx_vis, self.binned_cells_vis, self.peak_run = self.getSpeedTuning(binsize=800)
        self.sweeplength = self.stim_table.start.iloc[1] - self.stim_table.start.iloc[0]
        self.sweep_response = self.getSweepResponse()
        self.peak = self.getPeak(movie_name=movie_name)     
        
    def getSweepResponse(self):
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for index, row in self.stim_table.iterrows():
            start = row.start
            end = start + self.sweeplength
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.celltraces_dff[nc,start:end]
        return sweep_response
    
    def getPeak(self, movie_name):
        peak_movie = pd.DataFrame(index=range(self.numbercells), columns=('peak','response_variability'))

        for nc in range(self.numbercells):
            meanresponse = self.sweep_response[str(nc)].mean()
            movie_len= len(meanresponse)/30
            output = np.empty((movie_len,10))
            for tr in range(10):
                test = self.sweep_response[str(nc)].iloc[tr]
                for i in range(movie_len):
                    _,p = st.ks_2samp(test[i*30:(i+1)*30],test[(i+1)*30:(i+2)*30])
                    output[i,tr] = p    
            output = np.where(output<0.05, 1, 0)
            ptime = np.sum(output, axis=1)
            ptime*=10
            peak = np.argmax(meanresponse)
            if peak>30:
                peak_movie.response_variability.iloc[nc] = ptime[(peak-30)/30]
            else:
                peak_movie.response_variability.iloc[nc] = ptime[0]
            peak_movie.peak.iloc[nc] = peak
        if movie_name=='natural_movie_one':
            peak_movie.rename(columns={'peak':'peak_nm1','response_variability':'response_variability_nm1'}, inplace=True)
        elif movie_name=='natural_movie_two':
            peak_movie.rename(columns={'peak':'peak_nm2','response_variability':'response_variability_nm2'}, inplace=True)
        elif movie_name=='natural_movie_three':
            peak_movie.rename(columns={'peak':'peak_nm3','response_variability':'response_variability_nm3'}, inplace=True)

        return peak_movie

class LocallySN(OPAnalysis):    
    def __init__(self, cam_analysis, **kwargs):
        super(LocallySN, self).__init__(cam_analysis, **kwargs)        
        self.stim_table = self.cam_analysis.nwb.get_stimulus_table('locally_sparse_noise')
        self.LSN = self.cam_analysis.nwb.get_stimulus_template('locally_sparse_noise')
        self.sweeplength = self.stim_table['end'][1] - self.stim_table['start'][1]
        self.interlength = 4*self.sweeplength
        self.extralength = self.sweeplength
        self.sweep_response, self.mean_sweep_response, self.pval = self.getSweepResponse()
        self.receptive_field = self.getReceptiveField()
        
    def getReceptiveField(self):
        print "Calculating mean responses"
        receptive_field = np.empty((16, 28, self.numbercells+1, 2))

        for xp in range(16):
            for yp in range(28):
                on_frame = np.where(self.LSN[:,xp,yp]==255)[0]
                off_frame = np.where(self.LSN[:,xp,yp]==0)[0]
                subset_on = self.mean_sweep_response[self.stim_table.frame.isin(on_frame)]
                subset_off = self.mean_sweep_response[self.stim_table.frame.isin(off_frame)]
                receptive_field[xp,yp,:,0] = subset_on.mean(axis=0)
                receptive_field[xp,yp,:,1] = subset_off.mean(axis=0)
        return receptive_field  
