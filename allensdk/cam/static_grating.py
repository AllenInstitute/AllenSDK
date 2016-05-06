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
from allensdk.cam.cam_exceptions import CamAnalysisException


class StaticGrating(OPAnalysis):
    _log = logging.getLogger('allensdk.cam.static_grating')        
    
    def __init__(self, cam_analysis, **kwargs):
        super(StaticGrating, self).__init__(cam_analysis, **kwargs)
        
        stimulus_table = self.cam_analysis.nwb.get_static_gratings_stimulus_table()
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
        self.sweep_response, self.mean_sweep_response, self.pval = self.get_sweep_response()
        self.response = self.get_response()
        self.peak = self.get_peak()
        
    def get_response(self):
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
    
    
    def get_peak(self):    
        '''finds the peak response for each cell'''
        StaticGrating._log.info('Calculating peak response properties')

        peak = pd.DataFrame(index=range(self.numbercells), columns=('ori_sg','sf_sg', 'phase_sg', 'response_variability_sg','osi_sg','peak_dff_sg','ptest_sg','time_to_peak_sg','duration_sg'))

        for nc in range(self.numbercells):
            cell_peak = np.where(self.response[:,1:,:,nc,0] == np.nanmax(self.response[:,1:,:,nc,0]))
            pref_ori = cell_peak[0][0]
            pref_sf = cell_peak[1][0]+1
            pref_phase = cell_peak[2][0]
            peak.ori_sg[nc] = pref_ori
            peak.sf_sg[nc] = pref_sf
            peak.phase_sg[nc] = pref_phase
            peak.response_variability_sg[nc] = self.response[pref_ori, pref_sf, pref_phase, nc, 2]/0.48  #TODO: check number of trials
            pref = self.response[pref_ori, pref_sf, pref_phase, nc, 0]
            orth = self.response[np.mod(pref_ori+3, 6), pref_sf, pref_phase, nc, 0]
            peak.osi_sg[nc] = (pref-orth)/(pref+orth)
            peak.peak_dff_sg[nc] = pref
            groups = []
            
            for ori in self.orivals:
                for sf in self.sfvals[1:]:
                    for phase in self.phasevals:
                        groups.append(self.mean_sweep_response[(self.stim_table.spatial_frequency==sf)&(self.stim_table.orientation==ori)&(self.stim_table.phase==phase)][str(nc)])
            groups.append(self.mean_sweep_response[self.stim_table.spatial_frequency==0][str(nc)])

            _,p = st.f_oneway(*groups)
            peak.ptest_sg[nc] = p
            
            test_rows = (self.stim_table.orientation==self.orivals[pref_ori]) & \
                (self.stim_table.spatial_frequency==self.sfvals[pref_sf]) & \
                (self.stim_table.phase==self.phasevals[pref_phase])

            if len(test_rows) < 2:
                msg = "Static grating p value requires at least 2 trials at the preferred " 
                "orientation/spatial frequency/phase. Cell %d (%f, %f, %f) has %d." % \
                    (int(nc), self.orivals[pref_ori], self.sfvals[pref_sf], 
                     self.phasevals[pref_phase], len(test_rows))

                raise CamAnalysisException(msg)

            test = self.sweep_response[test_rows][str(nc)].mean()
            peak.time_to_peak_sg[nc] = (np.argmax(test) - self.interlength)/self.acquisition_rate
            test2 = np.where(test<(test.max()/2))[0]
            try:          
                peak.duration_sg[nc] = np.ediff1d(test2).max()/self.acquisition_rate
            except:
                pass

        return peak
