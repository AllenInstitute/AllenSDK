import scipy.stats as st
import numpy as np
import pandas as pd
from allensdk.cam.o_p_analysis import OPAnalysis

class NaturalImages(OPAnalysis):
    def __init__(self, cam_analysis, **kwargs):
        super(NaturalImages, self).__init__(cam_analysis, **kwargs)
        self.stim_table = self.cam_analysis.nwb.get_stimulus_table('natural_scenes')        
        self.number_images = len(np.unique(self.stim_table.frame))
        self.sweeplength = self.stim_table.end.iloc[1] - self.stim_table.start.iloc[1]
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
            subset_response = self.mean_sweep_response[self.stim_table.frame==(ni-1)]
            subset_pval = self.pval[self.stim_table.frame==(ni-1)]            
            response[ni,:,0] = subset_response.mean(axis=0)
            response[ni,:,1] = subset_response.std(axis=0)/np.sqrt(len(subset_response))
            response[ni,:,2] = subset_pval.apply(ptest, axis=0)
        return response
    
    def getPeak(self):    
        '''gets metrics about peak response, etc.'''
        print 'Calculating peak response properties'
        peak = pd.DataFrame(index=range(self.numbercells), columns=('Image', 'response_variability','peak_DFF', 'ptest', 'p_run', 'run_modulation', 'time_to_peak','duration'))
        peak['ExperimentID'] = self.experiment_id
        peak['Cre'] = self.Cre   
        peak['HVA'] = self.HVA
        peak['depth'] = self.cam_analysis.depth
        for nc in range(self.numbercells):
            nip = np.argmax(self.response[1:,nc,0])
            peak.Image[nc] = nip
            peak.response_variability[nc] = self.response[nip+1,nc,2]/0.50 #assume 50 trials
            peak.peak_DFF[nc] = self.response[nip+1,nc,0]
            subset = self.mean_sweep_response[self.stim_table.frame==nip]
#            blank = self.mean_sweep_response[self.stim_table.frame==-1]
            subset_stat = subset[subset.dx<2]
            subset_run = subset[subset.dx>=2]
            if (len(subset_run)>5) & ( len(subset_stat)>5):
                (_, peak.p_run[nc]) = st.ks_2samp(subset_run[str(nc)], subset_stat[str(nc)])
                peak.run_modulation[nc] = subset_run[str(nc)].mean()/subset_stat[str(nc)].mean()
            else:
                peak.p_run[nc] = np.NaN
                peak.run_modulation[nc] = np.NaN
            groups = []
            for im in range(self.number_images):
                subset = self.mean_sweep_response[self.stim_table.frame==(im-1)]
                groups.append(subset[str(nc)].values)
            (_,peak.ptest[nc]) = st.f_oneway(*groups)
            test = self.sweep_response[self.stim_table.frame==nip][str(nc)].mean()
            peak.time_to_peak[nc] = (np.argmax(test) - self.interlength)/self.acquisition_rate
            test2 = np.where(test<(test.max()/2))[0]          
            peak.duration[nc] = np.ediff1d(test2).max()/self.acquisition_rate

        return peak
