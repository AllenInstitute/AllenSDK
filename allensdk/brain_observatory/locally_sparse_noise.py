import scipy.stats as st
import pandas as pd
import numpy as np
from allensdk.brain_observatory.o_p_analysis import OPAnalysis

class LocallySparseNoise(OPAnalysis):    
    LSN_ON = 255
    LSN_OFF = 0
    LSN_GREY = 127
    LSN_OFF_SCREEN = 64

    def __init__(self, brain_observatory_analysis, **kwargs):
        super(LocallySparseNoise, self).__init__(brain_observatory_analysis, **kwargs)        
        self.stim_table = self.brain_observatory_analysis.nwb.get_locally_sparse_noise_stimulus_table()
        self.LSN, self.LSN_mask = self.brain_observatory_analysis.nwb.get_locally_sparse_noise_stimulus_template()
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
                on_frame = np.where(self.LSN[:,xp,yp]==self.LSN_ON)[0]
                off_frame = np.where(self.LSN[:,xp,yp]==self.LSN_OFF)[0]
                subset_on = self.mean_sweep_response[self.stim_table.frame.isin(on_frame)]
                subset_off = self.mean_sweep_response[self.stim_table.frame.isin(off_frame)]
                receptive_field[xp,yp,:,0] = subset_on.mean(axis=0)
                receptive_field[xp,yp,:,1] = subset_off.mean(axis=0)
        return receptive_field  
