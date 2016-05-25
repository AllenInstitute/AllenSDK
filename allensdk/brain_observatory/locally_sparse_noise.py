import scipy.stats as st
import pandas as pd
import numpy as np
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis

class LocallySparseNoise(StimulusAnalysis):    
    """ Perform tuning analysis specific to the locally sparse noise stimulus. 

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    
    nrows: int
       Number of rows in the stimulus template

    ncol: int
       Number of columns in the stimulus template
    """

    LSN_ON = 255
    LSN_OFF = 0
    LSN_GREY = 127
    LSN_OFF_SCREEN = 64

    def __init__(self, data_set, nrows=None, ncols=None, **kwargs):
        super(LocallySparseNoise, self).__init__(data_set, **kwargs)        
        self.nrows = 16 if nrows is None else nrows
        self.ncols = 28 if ncols is None else ncols

        self.stim_table = self.data_set.get_stimulus_table('locally_sparse_noise')
        self.LSN, self.LSN_mask = self.data_set.get_locally_sparse_noise_stimulus_template(mask_off_screen=False)
        self.sweeplength = self.stim_table['end'][1] - self.stim_table['start'][1]
        self.interlength = 4*self.sweeplength
        self.extralength = self.sweeplength
        self.sweep_response, self.mean_sweep_response, self.pval = self.get_sweep_response()
        self.receptive_field = self.get_receptive_field()

        
    def get_receptive_field(self):
        ''' Calculates receptive fields for each cell
        '''
        print "Calculating mean responses"
        receptive_field = np.empty((self.nrows, self.ncols, self.numbercells+1, 2))

        for xp in range(self.nrows):
            for yp in range(self.ncols):
                on_frame = np.where(self.LSN[:,xp,yp]==self.LSN_ON)[0]
                off_frame = np.where(self.LSN[:,xp,yp]==self.LSN_OFF)[0]
                subset_on = self.mean_sweep_response[self.stim_table.frame.isin(on_frame)]
                subset_off = self.mean_sweep_response[self.stim_table.frame.isin(off_frame)]
                receptive_field[xp,yp,:,0] = subset_on.mean(axis=0)
                receptive_field[xp,yp,:,1] = subset_off.mean(axis=0)
        return receptive_field  
