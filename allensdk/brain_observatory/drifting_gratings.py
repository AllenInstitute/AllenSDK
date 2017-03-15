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

from .stimulus_analysis import StimulusAnalysis
import scipy.stats as st
import pandas as pd
import numpy as np
import h5py
from math import sqrt
import logging
from . import observatory_plots as oplots
from . import circle_plots as cplots
from .brain_observatory_exceptions import MissingStimulusException
import matplotlib.pyplot as plt

class DriftingGratings(StimulusAnalysis):
    """ Perform tuning analysis specific to drifting gratings stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.drifting_gratings')

    def __init__(self, data_set, **kwargs):
        super(DriftingGratings, self).__init__(data_set, **kwargs)

        self.sweeplength = 60
        self.interlength = 30
        self.extralength = 0

        self._orivals = DriftingGratings._PRELOAD
        self._tfvals = DriftingGratings._PRELOAD
        self._number_ori = DriftingGratings._PRELOAD
        self._number_tf = DriftingGratings._PRELOAD

    @property
    def orivals(self):
        if self._orivals is DriftingGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._orivals

    @property
    def tfvals(self):
        if self._tfvals is DriftingGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._tfvals

    @property
    def number_ori(self):
        if self._number_ori is DriftingGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._number_ori

    @property
    def number_tf(self):
        if self._number_tf is DriftingGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._number_tf

    def populate_stimulus_table(self):
        stimulus_table = self.data_set.get_stimulus_table('drifting_gratings')
        self._stim_table = stimulus_table.fillna(value=0.)
        self._orivals = np.unique(self.stim_table.orientation).astype(int)
        self._tfvals = np.unique(self.stim_table.temporal_frequency).astype(int)
        self._number_ori = len(self.orivals)
        self._number_tf = len(self.tfvals)

    def get_response(self):
        ''' Computes the mean response for each cell to each stimulus condition.  Return is
        a (# orientations, # temporal frequencies, # cells, 3) np.ndarray.  The final dimension
        contains the mean response to the condition (index 0), standard error of the mean of the response
        to the condition (index 1), and the number of trials with a significant response (p < 0.05)
        to that condition (index 2).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        DriftingGratings._log.info("Calculating mean responses")

        response = np.empty(
            (self.number_ori, self.number_tf, self.numbercells + 1, 3))

        def ptest(x):
            return len(np.where(x < (0.05 / (8 * 5)))[0])

        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]
            for tf in self.tfvals:
                tf_pt = np.where(self.tfvals == tf)[0][0]
                subset_response = self.mean_sweep_response[
                    (self.stim_table.temporal_frequency == tf) & (self.stim_table.orientation == ori)]
                subset_pval = self.pval[(self.stim_table.temporal_frequency == tf) & (
                    self.stim_table.orientation == ori)]
                response[ori_pt, tf_pt, :, 0] = subset_response.mean(axis=0)
                response[ori_pt, tf_pt, :, 1] = subset_response.std(
                    axis=0) / sqrt(len(subset_response))
                response[ori_pt, tf_pt, :, 2] = subset_pval.apply(
                    ptest, axis=0)
        return response

    def get_peak(self):
        ''' Computes metrics related to each cell's peak response condition.

        Returns
        -------
        Pandas data frame containing the following columns (_dg suffix is
        for drifting grating):
            * ori_dg (orientation)
            * tf_dg (temporal frequency)
            * response_reliability_dg
            * osi_dg (orientation selectivity index)
            * dsi_dg (direction selectivity index)
            * peak_dff_dg (peak dF/F)
            * ptest_dg
            * p_run_dg
            * run_modulation_dg
            * cv_dg (circular variance)
        '''
        DriftingGratings._log.info('Calculating peak response properties')

        peak = pd.DataFrame(index=range(self.numbercells), columns=('ori_dg', 'tf_dg', 'response_reliability_dg',
                                                                    'osi_dg', 'dsi_dg', 'peak_dff_dg', 'ptest_dg', 'p_run_dg', 'run_modulation_dg', 'cv_dg', 'cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        orivals_rad = np.deg2rad(self.orivals)
        for nc in range(self.numbercells):
            cell_peak = np.where(self.response[:, 1:, nc, 0] == np.nanmax(
                self.response[:, 1:, nc, 0]))
            prefori = cell_peak[0][0]
            preftf = cell_peak[1][0] + 1
            peak.cell_specimen_id.iloc[nc] = cids[nc]
            peak.ori_dg.iloc[nc] = prefori
            peak.tf_dg.iloc[nc] = preftf
            peak.response_reliability_dg.iloc[
                nc] = self.response[prefori, preftf, nc, 2] / 0.15
            pref = self.response[prefori, preftf, nc, 0]
            orth1 = self.response[np.mod(prefori + 2, 8), preftf, nc, 0]
            orth2 = self.response[np.mod(prefori - 2, 8), preftf, nc, 0]
            orth = (orth1 + orth2) / 2
            null = self.response[np.mod(prefori + 4, 8), preftf, nc, 0]

            tuning = self.response[:, preftf, nc, 0]
            CV_top = np.empty((8))
            for i in range(8):
                CV_top[i] = (tuning[i] * np.exp(1j * 2 * orivals_rad[i])).real
            peak.cv_dg.iloc[nc] = np.abs(CV_top.sum() / tuning.sum())

            peak.osi_dg.iloc[nc] = (pref - orth) / (pref + orth)
            peak.dsi_dg.iloc[nc] = (pref - null) / (pref + null)
            peak.peak_dff_dg.iloc[nc] = pref

            groups = []
            for ori in self.orivals:
                for tf in self.tfvals[1:]:
                    groups.append(self.mean_sweep_response[(self.stim_table.temporal_frequency == tf) & (
                        self.stim_table.orientation == ori)][str(nc)])
            groups.append(self.mean_sweep_response[
                          self.stim_table.temporal_frequency == 0][str(nc)])
            _, p = st.f_oneway(*groups)
            peak.ptest_dg.iloc[nc] = p

            subset = self.mean_sweep_response[(self.stim_table.temporal_frequency == self.tfvals[
                                               preftf]) & (self.stim_table.orientation == self.orivals[prefori])]
            subset_stat = subset[subset.dx < 1]
            subset_run = subset[subset.dx >= 1]
            if (len(subset_run) > 2) & (len(subset_stat) > 2):
                (_, peak.p_run_dg.iloc[nc]) = st.ks_2samp(
                    subset_run[str(nc)], subset_stat[str(nc)])
                peak.run_modulation_dg.iloc[nc] = subset_run[
                    str(nc)].mean() / subset_stat[str(nc)].mean()
            else:
                peak.p_run_dg.iloc[nc] = np.NaN
                peak.run_modulation_dg.iloc[nc] = np.NaN

        return peak

    def open_star_plot(self, cell_specimen_id, include_labels=False):
        cell_id = self.peak_row_from_csid(self.peak, cell_specimen_id)

        df = self.mean_sweep_response[str(cell_id)]
        st = self.data_set.get_stimulus_table('drifting_gratings')
        mask = st.dropna(subset=['orientation']).index
        
        data = df.values
    
        cmin = self.response[0,0,cell_id,0]
        cmax = data.mean() + data.std()*3

        fp = cplots.FanPlotter.for_drifting_gratings()
        fp.plot(r_data=st.temporal_frequency.ix[mask].values,
                angle_data=st.orientation.ix[mask].values,
                data=df.ix[mask].values,
                clim=[cmin, cmax])
        fp.show_axes(closed=True)
    
        if include_labels:
            fp.show_r_labels()
            fp.show_angle_labels()

    def plot_orientation_selectivity(self,
                                     si_range=oplots.SI_RANGE,
                                     n_hist_bins=oplots.N_HIST_BINS,
                                     color=oplots.STIM_COLOR,
                                     p_value_max=oplots.P_VALUE_MAX,
                                     peak_dff_min=oplots.PEAK_DFF_MIN):
        # responsive cells 
        vis_cells = (self.peak.ptest_dg < p_value_max) & (self.peak.peak_dff_dg > peak_dff_min)

        # orientation selective cells
        osi_cells = vis_cells & (self.peak.osi_dg > si_range[0]) & (self.peak.osi_dg < si_range[1])

        peak_osi = self.peak.ix[osi_cells]
        osis = peak_osi.osi_dg.values

        oplots.plot_selectivity_cumulative_histogram(osis, 
                                                     "orientation selectivity index",
                                                     si_range=si_range,
                                                     n_hist_bins=n_hist_bins,
                                                     color=color)

    def plot_direction_selectivity(self,
                                   si_range=oplots.SI_RANGE,
                                   n_hist_bins=oplots.N_HIST_BINS,
                                   color=oplots.STIM_COLOR,
                                   p_value_max=oplots.P_VALUE_MAX,
                                   peak_dff_min=oplots.PEAK_DFF_MIN):

        # responsive cells 
        vis_cells = (self.peak.ptest_dg < p_value_max) & (self.peak.peak_dff_dg > peak_dff_min)

        # direction selective cells
        dsi_cells = vis_cells & (self.peak.dsi_dg > si_range[0]) & (self.peak.dsi_dg < si_range[1])

        peak_dsi = self.peak.ix[dsi_cells]
        dsis = peak_dsi.dsi_dg.values

        oplots.plot_selectivity_cumulative_histogram(dsis, 
                                                     "direction selectivity index",
                                                     si_range=si_range,
                                                     n_hist_bins=n_hist_bins,
                                                     color=color)

    def plot_preferred_direction(self,
                                 include_labels=False,
                                 si_range=oplots.SI_RANGE,
                                 color=oplots.STIM_COLOR,
                                 p_value_max=oplots.P_VALUE_MAX,
                                 peak_dff_min=oplots.PEAK_DFF_MIN):
        vis_cells = (self.peak.ptest_dg < p_value_max) & (self.peak.peak_dff_dg > peak_dff_min)    
        pref_dirs = self.peak.ix[vis_cells].ori_dg.values
        pref_dirs = [ self.orivals[pref_dir] for pref_dir in pref_dirs ]

        angles, counts = np.unique(pref_dirs, return_counts=True)
        oplots.plot_radial_histogram(angles, 
                                     counts,
                                     include_labels=include_labels,
                                     all_angles=self.orivals,
                                     direction=-1,
                                     offset=0.0,
                                     closed=True,
                                     color=color)

    def plot_preferred_temporal_frequency(self, 
                                          si_range=oplots.SI_RANGE,
                                          color=oplots.STIM_COLOR,
                                          p_value_max=oplots.P_VALUE_MAX,
                                          peak_dff_min=oplots.PEAK_DFF_MIN):

        vis_cells = (self.peak.ptest_dg < p_value_max) & (self.peak.peak_dff_dg > peak_dff_min)    
        pref_tfs = self.peak.ix[vis_cells].tf_dg.values

        oplots.plot_condition_histogram(pref_tfs, 
                                        self.tfvals[1:],
                                        color=color)

        plt.xlabel("temporal frequency (Hz)")
        plt.ylabel("number of cells")


    @staticmethod
    def from_analysis_file(data_set, analysis_file):
        dg = DriftingGratings(data_set)

        try:
            dg.populate_stimulus_table()

            dg._sweep_response = pd.read_hdf(analysis_file, "analysis/sweep_response_dg")
            dg._mean_sweep_response = pd.read_hdf(analysis_file, "analysis/mean_sweep_response_dg")
            dg._peak = pd.read_hdf(analysis_file, "analysis/peak")

            with h5py.File(analysis_file, "r") as f:
                dg._response = f["analysis/response_dg"].value
                dg._binned_dx_sp = f["analysis/binned_dx_sp"].value
                dg._binned_cells_sp = f["analysis/binned_cells_sp"].value
                dg._binned_dx_vis = f["analysis/binned_dx_vis"].value
                dg._binned_cells_vis = f["analysis/binned_cells_vis"].value
        except Exception as e:
            raise MissingStimulusException(e.args)

        return dg

