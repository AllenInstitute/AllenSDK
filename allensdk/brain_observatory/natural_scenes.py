# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import scipy.stats as st
import numpy as np
import pandas as pd
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
import logging

class NaturalScenes(StimulusAnalysis):
    """ Perform tuning analysis specific to natural scenes stimulus. 

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.natural_scenes')    
    
    def __init__(self, data_set, **kwargs):
        super(NaturalScenes, self).__init__(data_set, **kwargs)
        self.stim_table = self.data_set.get_stimulus_table('natural_scenes')
        self.number_scenes = len(np.unique(self.stim_table.frame))
        self.sweeplength = self.stim_table.end.iloc[1] - self.stim_table.start.iloc[1]
        self.interlength = 4 * self.sweeplength
        self.extralength = self.sweeplength
        self.sweep_response, self.mean_sweep_response, self.pval = self.get_sweep_response()
        self.response = self.get_response()
        self.peak = self.get_peak()
        
    def get_response(self):
        ''' Computes the mean response for each cell to each stimulus condition.  Return is 
        a (# scenes, # cells, 3) np.ndarray.  The final dimension
        contains the mean response to the condition (index 0), standard error of the mean of the response
        to the condition (index 1), and p value of the response to that condition (index 3).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        NaturalScenes._log.info("Calculating mean responses")
        
        response = np.empty((self.number_scenes, self.numbercells+1, 3))
        
        def ptest(x):
            return len(np.where(x<(0.05/(self.number_scenes-1)))[0])
            
        for ns in range(self.number_scenes):
            subset_response = self.mean_sweep_response[self.stim_table.frame==(ns-1)]
            subset_pval = self.pval[self.stim_table.frame==(ns-1)]            
            response[ns,:,0] = subset_response.mean(axis=0)
            response[ns,:,1] = subset_response.std(axis=0)/np.sqrt(len(subset_response))
            response[ns,:,2] = subset_pval.apply(ptest, axis=0)
        
        return response
    
    def get_peak(self):    
        ''' Computes metrics about peak response condition for each cell.
        
        Returns
        -------
        Pandas data frame with the following fields ('_ns' suffix is for
        natural scene):
            * scene_ns (scene number)
            * response_reliability_ns
            * peak_dff_ns (peak dF/F)
            * ptest_ns
            * p_run_ns
            * run_modulation_ns
            * time_to_peak_ns
            * duration_ns
        '''
        NaturalScenes._log.info('Calculating peak response properties')
        peak = pd.DataFrame(index=range(self.numbercells), columns=('scene_ns', 'response_reliability_ns','peak_dff_ns', 'ptest_ns', 'p_run_ns', 'run_modulation_ns', 'time_to_peak_ns','duration_ns','cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        for nc in range(self.numbercells):
            nsp = np.argmax(self.response[1:,nc,0])
            peak.cell_specimen_id.iloc[nc] = cids[nc]
            peak.scene_ns[nc] = nsp
            peak.response_reliability_ns[nc] = self.response[nsp+1,nc,2]/0.50 #assume 50 trials
            peak.peak_dff_ns[nc] = self.response[nsp+1,nc,0]
            subset = self.mean_sweep_response[self.stim_table.frame==nsp]
            subset_stat = subset[subset.dx<2]
            subset_run = subset[subset.dx>=2]
            if (len(subset_run)>5) & ( len(subset_stat)>5):
                (_, peak.p_run_ns[nc]) = st.ks_2samp(subset_run[str(nc)], subset_stat[str(nc)])
                peak.run_modulation_ns[nc] = subset_run[str(nc)].mean()/subset_stat[str(nc)].mean()
            else:
                peak.p_run_ns[nc] = np.NaN
                peak.run_modulation_ns[nc] = np.NaN
            groups = []
            for im in range(self.number_scenes):
                subset = self.mean_sweep_response[self.stim_table.frame==(im-1)]
                groups.append(subset[str(nc)].values)
            (_,peak.ptest_ns[nc]) = st.f_oneway(*groups)
            test = self.sweep_response[self.stim_table.frame==nsp][str(nc)].mean()
            peak.time_to_peak_ns[nc] = (np.argmax(test) - self.interlength)/self.acquisition_rate
            test2 = np.where(test<(test.max()/2))[0]
            try:          
                peak.duration_ns[nc] = np.ediff1d(test2).max()/self.acquisition_rate
            except:
                pass

        return peak
