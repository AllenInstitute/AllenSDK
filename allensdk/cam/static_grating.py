import scipy.stats as st
import numpy as np
import pandas as pd
from math import sqrt
from allensdk.cam.o_p_analysis import OPAnalysis


class StaticGrating(OPAnalysis):
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
        #self.binned_dx_sp, self.binned_cells_sp, self.binned_dx_vis, self.binned_cells_vis = self.getSpeedTuning(binsize=200)
        
#    
    def getResponse(self):
#        if self.h5path != None:
#            response = op.loadh5(self.h5path, 'response')
        print "Calculating mean responses"
        response = np.empty((self.number_ori, self.number_sf, self.number_phase, self.numbercells+1, 3))
#        blank = np.empty((self.numbercells, 3))        
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
#        subset_response = self.mean_sweep_response[self.stim_table.orientation.isnull()]
#        subset_pval = self.pval[self.stim_table.orientation.isnull()]
#        blank[:,0] = subset_response.mean(axis=0)
#        blank[:,1] = subset_response.std(axis=0)/sqrt(len(subset_response))
#        blank[:,2] = subset_pval.apply(ptest, axis=0)
        return response#, blank
    
    def getPeak(self):    
        '''finds the peak response for each cell'''
        print 'Calculating peak response properties'
        peak = pd.DataFrame(index=range(self.numbercells), columns=('Ori','SF', 'Phase', 'sg_response_variability','OSI','sg_peak_DFF','ptest'))
        peak['ExperimentID'] = self.experiment_id
        peak['Cre'] = self.Cre   
        peak['HVA'] = self.HVA
        peak['depth'] = self.depth

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
            #TODO: add ptest, reliability, etc
#        peak['ptest'] = self.ptest
        return peak
    
#    def Ptest(self):
#        '''running new ptest'''
#        test = pd.DataFrame(index=self.sweeptable.index.values, columns=np.array(range(self.numbercells)).astype(str))
#        for nc in range(self.numbercells):        
#            for index, row in self.sweeptable.iterrows():
#                ori=row.Ori
#                sf=row.SF
#                phase = row.Phase
#                test[str(nc)][index] = self.mean_sweep_response[(self.stim_table.spatial_frequency==sf)&(self.stim_table.orientation==ori)&(self.stim_table.phase==phase)][str(nc)]
#        ptest = []
#        for nc in range(self.numbercells):
#            groups = []
#            for index,row in test.iterrows():
#                groups.append(test[str(nc)][index])
#                (f,p) = st.f_oneway(*groups)
#            ptest.append(p)
#        ptest = np.array(ptest)
#        cells = list(np.where(ptest<0.01)[0])
#        print "# cells: " + str(len(ptest))
#        print "# significant cells: " + str(len(cells))
#        return ptest, cells
