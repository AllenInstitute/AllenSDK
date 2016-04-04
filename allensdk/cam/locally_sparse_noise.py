import scipy.stats as st
import pandas as pd
import numpy as np
from allensdk.cam.o_p_analysis import OPAnalysis

class LocallySparseNoise(OPAnalysis):    
    def __init__(self, cam_analysis, **kwargs):
        super(LocallySN, self).__init__(cam_analysis, **kwargs)        
        self.stim_table = self.cam_analysis.nwb.get_stimulus_table('locally_sparse_noise')
        self.LSN = self.cam_analysis.nwb.get_stimulus_template('locally_sparse_noise')
        self.sweeplength = self.stim_table['end'][1] - self.stim_table['start'][1]
        self.interlength = 4*self.sweeplength
        self.extralength = self.sweeplength
        self.sweep_response, self.mean_sweep_response, self.pval = self.get_sweep_response()
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
