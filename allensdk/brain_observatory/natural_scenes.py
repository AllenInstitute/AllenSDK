# Copyright 2016-2017 Allen Institute for Brain Science
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
from .stimulus_analysis import StimulusAnalysis
import logging
import h5py
from . import observatory_plots as oplots
from . import circle_plots as cplots
from .brain_observatory_exceptions import MissingStimulusException

class NaturalScenes(StimulusAnalysis):
    """ Perform tuning analysis specific to natural scenes stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.natural_scenes')

    def __init__(self, data_set, **kwargs):
        super(NaturalScenes, self).__init__(data_set, **kwargs)

        self._number_scenes = StimulusAnalysis._PRELOAD
        self._sweeplength = StimulusAnalysis._PRELOAD
        self._interlength = StimulusAnalysis._PRELOAD
        self._extralength = StimulusAnalysis._PRELOAD

    @property
    def number_scenes(self):
        if self._number_scenes is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._number_scenes

    @property
    def sweeplength(self):
        if self._sweeplength is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._sweeplength

    @property
    def interlength(self):
        if self._interlength is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._interlength

    @property
    def extralength(self):
        if self._extralength is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._extralength

    def populate_stimulus_table(self):
        self._stim_table = self.data_set.get_stimulus_table('natural_scenes')
        self._number_scenes = len(np.unique(self._stim_table.frame))
        self._sweeplength = self._stim_table.end.iloc[
            1] - self._stim_table.start.iloc[1]
        self._interlength = 4 * self._sweeplength
        self._extralength = self._sweeplength

    def get_response(self):
        ''' Computes the mean response for each cell to each stimulus condition.  Return is
        a (# scenes, # cells, 3) np.ndarray.  The final dimension
        contains the mean response to the condition (index 0), standard error of the mean of the response
        to the condition (index 1), and the number of trials with a significant (p < 0.05) response 
        to that condition (index 2).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        NaturalScenes._log.info("Calculating mean responses")

        response = np.empty((self.number_scenes, self.numbercells + 1, 3))

        def ptest(x):
            return len(np.where(x < (0.05 / (self.number_scenes - 1)))[0])

        for ns in range(self.number_scenes):
            subset_response = self.mean_sweep_response[
                self.stim_table.frame == (ns - 1)]
            subset_pval = self.pval[self.stim_table.frame == (ns - 1)]
            response[ns, :, 0] = subset_response.mean(axis=0)
            response[ns, :, 1] = subset_response.std(
                axis=0) / np.sqrt(len(subset_response))
            response[ns, :, 2] = subset_pval.apply(ptest, axis=0)

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
        peak = pd.DataFrame(index=range(self.numbercells), columns=('scene_ns', 'response_reliability_ns', 'peak_dff_ns',
                                                                    'ptest_ns', 'p_run_ns', 'run_modulation_ns', 'time_to_peak_ns', 'duration_ns', 'cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        for nc in range(self.numbercells):
            nsp = np.argmax(self.response[1:, nc, 0])
            peak.cell_specimen_id.iloc[nc] = cids[nc]
            peak.scene_ns[nc] = nsp
            peak.response_reliability_ns[nc] = self.response[
                nsp + 1, nc, 2] / 0.50  # assume 50 trials
            peak.peak_dff_ns[nc] = self.response[nsp + 1, nc, 0]
            subset = self.mean_sweep_response[self.stim_table.frame == nsp]
            subset_stat = subset[subset.dx < 2]
            subset_run = subset[subset.dx >= 2]
            if (len(subset_run) > 5) & (len(subset_stat) > 5):
                (_, peak.p_run_ns[nc]) = st.ks_2samp(
                    subset_run[str(nc)], subset_stat[str(nc)])
                peak.run_modulation_ns[nc] = subset_run[
                    str(nc)].mean() / subset_stat[str(nc)].mean()
            else:
                peak.p_run_ns[nc] = np.NaN
                peak.run_modulation_ns[nc] = np.NaN
            groups = []
            for im in range(self.number_scenes):
                subset = self.mean_sweep_response[
                    self.stim_table.frame == (im - 1)]
                groups.append(subset[str(nc)].values)
            (_, peak.ptest_ns[nc]) = st.f_oneway(*groups)
            test = self.sweep_response[
                self.stim_table.frame == nsp][str(nc)].mean()
            peak.time_to_peak_ns[nc] = (
                np.argmax(test) - self.interlength) / self.acquisition_rate
            test2 = np.where(test < (test.max() / 2))[0]
            try:
                peak.duration_ns[nc] = np.ediff1d(
                    test2).max() / self.acquisition_rate
            except:
                pass

        return peak

    def plot_time_to_peak(self, 
                          p_value_max=oplots.P_VALUE_MAX, 
                          color_map=oplots.STIMULUS_COLOR_MAP):
        stimulus_table = self.data_set.get_stimulus_table('natural_scenes')

        resps = []

        for index, row in self.peak.iterrows():    
            mean_response = self.sweep_response.ix[stimulus_table.frame==row.scene_ns][str(index)].mean()
            resps.append((mean_response - mean_response.mean() / mean_response.std()))

        mean_responses = np.array(resps)

        sorted_table = self.peak[self.peak.ptest_ns < p_value_max].sort(columns='time_to_peak_ns')
        cell_order = sorted_table.index

        # time to peak is relative to stimulus start in seconds
        ttps = sorted_table.time_to_peak_ns.values + self.interlength / self.acquisition_rate
        msrs_sorted = mean_responses[cell_order,:]

        oplots.plot_time_to_peak(msrs_sorted, ttps,
                                 0, (2*self.interlength + self.sweeplength) / self.acquisition_rate,
                                 (self.interlength) / self.acquisition_rate, 
                                 (self.interlength + self.sweeplength) / self.acquisition_rate, 
                                 color_map)

    def open_corona_plot(self, cell_specimen_id):
        cell_id = self.peak_row_from_csid(self.peak, cell_specimen_id)

        df = self.mean_sweep_response[str(cell_id)]
        data = df.values

        st = self.data_set.get_stimulus_table('natural_scenes')
        mask = st[st.frame >= 0].index

        cmin = self.response[0,cell_id,0]
        cmax = data.mean() + data.std()*3

        cp = cplots.CoronaPlotter()
        cp.plot(st.frame.ix[mask].values, 
                data=df.ix[mask].values,
                clim=[cmin, cmax])
        cp.show_arrow()

    
    @staticmethod
    def from_analysis_file(data_set, analysis_file):
        ns = NaturalScenes(data_set)
        ns.populate_stimulus_table()

        try:
            ns._sweep_response = pd.read_hdf(analysis_file, "analysis/sweep_response_ns")
            ns._mean_sweep_response = pd.read_hdf(analysis_file, "analysis/mean_sweep_response_ns")
            ns._peak = pd.read_hdf(analysis_file, "analysis/peak")

            with h5py.File(analysis_file, "r") as f:
                ns._response = f["analysis/response_ns"].value
                ns._binned_dx_sp = f["analysis/binned_dx_sp"].value
                ns._binned_cells_sp = f["analysis/binned_cells_sp"].value
                ns._binned_dx_vis = f["analysis/binned_dx_vis"].value
                ns._binned_cells_vis = f["analysis/binned_cells_vis"].value
        except Exception as e:
            raise MissingStimulusException(e.args)

        return ns

