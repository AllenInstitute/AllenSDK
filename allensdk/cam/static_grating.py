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
from math import sqrt
import logging
from allensdk.cam.o_p_analysis import OPAnalysis


class StaticGrating(OPAnalysis):
    _log = logging.getLogger('allensdk.cam.static_grating')        
    
    def __init__(self, cam_analysis, **kwargs):
        super(StaticGrating, self).__init__(cam_analysis, **kwargs)
        
        stimulus_table = self.cam_analysis.nwb.get_stimulus_table('static_gratings')
        self.stim_table = stimulus_table.fillna(value=0.)     
        self.sweeplength = self.stim_table['end'].iloc[1] - self.stim_table['start'].iloc[1]
        self.interlength = 4 * self.sweeplength
        self.extralength = self.sweeplength
        self.orivals = np.unique(self.stim_table.orientation.dropna())
        self.sfvals = np.unique(self.stim_table.spatial_frequency.dropna())
        self.phasevals = np.unique(self.stim_table.phase.dropna())
        self.number_ori = len(self.orivals)
        self.number_sf = len(self.sfvals)
        self.number_phase = len(self.phasevals)            
        self.sweep_response, self.mean_sweep_response, self.pval = self.getSweepResponse()
        self.response = self.getResponse()
        self.peak = self.getPeak()

        
    def getResponse(self):
        StaticGrating._log.info("Calculating mean responses")
        
        response = np.empty((self.number_ori, self.number_sf, self.number_phase, self.numbercells+1, 3))

        
        def ptest(x):
            return len(np.where(x<(0.05/(self.number_ori*(self.number_sf-1))))[0])
            
        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]
            
            for sf in self.sfvals:
                sf_pt = np.where(self.sfvals == sf)[0][0]
                
                for phase in self.phasevals:
                    phase_pt = np.where(self.phasevals == phase)[0][0]
                    subset_response = self.mean_sweep_response[(self.stim_table.spatial_frequency==sf)&(self.stim_table.orientation==ori)&(self.stim_table.phase==phase)]
                    subset_pval = self.pval[(self.stim_table.spatial_frequency==sf)&(self.stim_table.orientation==ori)&(self.stim_table.phase==phase)]
                    response[ori_pt, sf_pt, phase_pt, :, 0] = subset_response.mean(axis=0)
                    response[ori_pt, sf_pt, phase_pt, :, 1] = subset_response.std(axis=0)/sqrt(len(subset_response))
                    response[ori_pt, sf_pt, phase_pt, :, 2] = subset_pval.apply(ptest, axis=0)

        return response
    
    
    def getPeak(self):    
        '''finds the peak response for each cell'''
        StaticGrating._log.info('Calculating peak response properties')
        
        peak = pd.DataFrame(index=range(self.numbercells),
                            columns=('Ori','SF', 'Phase', 'sg_response_variability','OSI','sg_peak_DFF','ptest'))

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
                        groups.append(self.mean_sweep_response[(self.stim_table.spatial_frequency==sf)&(self.stim_table.orientation==ori)&(self.stim_table.phase==phase)][str(nc)])
            
            _,p = st.f_oneway(*groups)
            peak.ptest[nc] = p

        return peak
