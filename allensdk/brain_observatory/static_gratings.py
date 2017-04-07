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
from .stimulus_analysis import StimulusAnalysis
from .brain_observatory_exceptions import BrainObservatoryAnalysisException, MissingStimulusException
from . import observatory_plots as oplots
from . import circle_plots as cplots
import h5py
import matplotlib.pyplot as plt

class StaticGratings(StimulusAnalysis):
    """ Perform tuning analysis specific to static gratings stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.static_gratings')

    def __init__(self, data_set, **kwargs):
        super(StaticGratings, self).__init__(data_set, **kwargs)

        self._sweeplength = StaticGratings._PRELOAD
        self._interlength = StaticGratings._PRELOAD
        self._extralength = StaticGratings._PRELOAD
        self._orivals = StaticGratings._PRELOAD
        self._sfvals = StaticGratings._PRELOAD
        self._phasevals = StaticGratings._PRELOAD
        self._number_ori = StaticGratings._PRELOAD
        self._number_sf = StaticGratings._PRELOAD
        self._number_phase = StaticGratings._PRELOAD

    @property
    def sweeplength(self):
        if self._sweeplength is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._sweeplength

    @property
    def interlength(self):
        if self._interlength is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._interlength

    @property
    def extralength(self):
        if self._extralength is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._extralength

    @property
    def orivals(self):
        if self._orivals is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._orivals

    @property
    def sfvals(self):
        if self._sfvals is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._sfvals

    @property
    def phasevals(self):
        if self._phasevals is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._phasevals

    @property
    def number_ori(self):
        if self._number_ori is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._number_ori

    @property
    def number_sf(self):
        if self._number_sf is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._number_sf

    @property
    def number_phase(self):
        if self._number_phase is StaticGratings._PRELOAD:
            self.populate_stimulus_table()

        return self._number_phase

    def populate_stimulus_table(self):
        stimulus_table = self.data_set.get_stimulus_table('static_gratings')
        self._stim_table = stimulus_table.fillna(value=0.)
        self._sweeplength = self.stim_table['end'].iloc[
            1] - self.stim_table['start'].iloc[1]
        self._interlength = 4 * self._sweeplength
        self._extralength = self._sweeplength
        self._orivals = np.unique(self._stim_table.orientation.dropna())
        self._sfvals = np.unique(self._stim_table.spatial_frequency.dropna())
        self._phasevals = np.unique(self._stim_table.phase.dropna())
        self._number_ori = len(self._orivals)
        self._number_sf = len(self._sfvals)
        self._number_phase = len(self._phasevals)

    def get_response(self):
        ''' Computes the mean response for each cell to each stimulus condition.  Return is
        a (# orientations, # spatial frequencies, # phasees, # cells, 3) np.ndarray.  The final dimension
        contains the mean response to the condition (index 0), standard error of the mean of the response
        to the condition (index 1), and the number of trials with a significant response (p < 0.05)
        to that condition (index 2).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        StaticGratings._log.info("Calculating mean responses")

        response = np.empty((self.number_ori, self.number_sf,
                             self.number_phase, self.numbercells + 1, 3))

        def ptest(x):
            return len(np.where(x < (0.05 / (self.number_ori * (self.number_sf - 1))))[0])

        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]

            for sf in self.sfvals:
                sf_pt = np.where(self.sfvals == sf)[0][0]

                for phase in self.phasevals:
                    phase_pt = np.where(self.phasevals == phase)[0][0]
                    subset_response = self.mean_sweep_response[(self.stim_table.spatial_frequency == sf) & (
                        self.stim_table.orientation == ori) & (self.stim_table.phase == phase)]
                    subset_pval = self.pval[(self.stim_table.spatial_frequency == sf) & (
                        self.stim_table.orientation == ori) & (self.stim_table.phase == phase)]
                    response[ori_pt, sf_pt, phase_pt, :,
                             0] = subset_response.mean(axis=0)
                    response[ori_pt, sf_pt, phase_pt, :, 1] = subset_response.std(
                        axis=0) / sqrt(len(subset_response))
                    response[ori_pt, sf_pt, phase_pt, :,
                             2] = subset_pval.apply(ptest, axis=0)

        return response

    def get_peak(self):
        ''' Computes metrics related to each cell's peak response condition.

        Returns
        -------
        Panda data frame with the following fields (_sg suffix is
        for static grating):
            * ori_sg (orientation)
            * sf_sg (spatial frequency)
            * phase_sg
            * response_variability_sg
            * osi_sg (orientation selectivity index)
            * peak_dff_sg (peak dF/F)
            * ptest_sg
            * time_to_peak_sg
            * duration_sg
        '''
        StaticGratings._log.info('Calculating peak response properties')

        peak = pd.DataFrame(index=range(self.numbercells), columns=('ori_sg', 'sf_sg', 'phase_sg', 'response_reliability_sg',
                                                                    'osi_sg', 'peak_dff_sg', 'ptest_sg', 'time_to_peak_sg', 'duration_sg', 'cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        for nc in range(self.numbercells):
            cell_peak = np.where(self.response[:, 1:, :, nc, 0] == np.nanmax(
                self.response[:, 1:, :, nc, 0]))
            pref_ori = cell_peak[0][0]
            pref_sf = cell_peak[1][0] + 1
            pref_phase = cell_peak[2][0]
            peak.cell_specimen_id.iloc[nc] = cids[nc]
            peak.ori_sg[nc] = pref_ori
            peak.sf_sg[nc] = pref_sf
            peak.phase_sg[nc] = pref_phase
            peak.response_reliability_sg[nc] = self.response[
                pref_ori, pref_sf, pref_phase, nc, 2] / 0.48  # TODO: check number of trials
            pref = self.response[pref_ori, pref_sf, pref_phase, nc, 0]
            orth = self.response[
                np.mod(pref_ori + 3, 6), pref_sf, pref_phase, nc, 0]
            peak.osi_sg[nc] = (pref - orth) / (pref + orth)
            peak.peak_dff_sg[nc] = pref
            groups = []

            for ori in self.orivals:
                for sf in self.sfvals[1:]:
                    for phase in self.phasevals:
                        groups.append(self.mean_sweep_response[(self.stim_table.spatial_frequency == sf) & (
                            self.stim_table.orientation == ori) & (self.stim_table.phase == phase)][str(nc)])
            groups.append(self.mean_sweep_response[
                          self.stim_table.spatial_frequency == 0][str(nc)])

            _, p = st.f_oneway(*groups)
            peak.ptest_sg[nc] = p

            test_rows = (self.stim_table.orientation == self.orivals[pref_ori]) & \
                (self.stim_table.spatial_frequency == self.sfvals[pref_sf]) & \
                (self.stim_table.phase == self.phasevals[pref_phase])

            if len(test_rows) < 2:
                msg = "Static grating p value requires at least 2 trials at the preferred "
                "orientation/spatial frequency/phase. Cell %d (%f, %f, %f) has %d." % \
                    (int(nc), self.orivals[pref_ori], self.sfvals[pref_sf],
                     self.phasevals[pref_phase], len(test_rows))

                raise BrainObservatoryAnalysisException(msg)

            test = self.sweep_response[test_rows][str(nc)].mean()
            peak.time_to_peak_sg[nc] = (
                np.argmax(test) - self.interlength) / self.acquisition_rate
            test2 = np.where(test < (test.max() / 2))[0]
            try:
                peak.duration_sg[nc] = np.ediff1d(
                    test2).max() / self.acquisition_rate
            except:
                pass

        return peak

    def plot_time_to_peak(self, 
                          p_value_max=oplots.P_VALUE_MAX, 
                          color_map=oplots.STIMULUS_COLOR_MAP):
        stimulus_table = self.data_set.get_stimulus_table('static_gratings')

        resps = []

        for index, row in self.peak.iterrows():
            pref_rows = (stimulus_table.orientation==self.orivals[row.ori_sg]) & \
                (stimulus_table.spatial_frequency==self.sfvals[row.sf_sg]) & \
                (stimulus_table.phase==self.phasevals[row.phase_sg])

            mean_response = self.sweep_response[pref_rows][str(index)].mean()
            resps.append((mean_response - mean_response.mean() / mean_response.std()))

        mean_responses = np.array(resps)

        sorted_table = self.peak[self.peak.ptest_sg < p_value_max].sort(columns='time_to_peak_sg')
        cell_order = sorted_table.index

        # time to peak is relative to stimulus start in seconds
        ttps = sorted_table.time_to_peak_sg.values + self.interlength / self.acquisition_rate
        msrs_sorted = mean_responses[cell_order,:]

        oplots.plot_time_to_peak(msrs_sorted, ttps,
                                 0, (2*self.interlength + self.sweeplength) / self.acquisition_rate,
                                 (self.interlength) / self.acquisition_rate, 
                                 (self.interlength + self.sweeplength) / self.acquisition_rate, 
                                 color_map)


    def plot_orientation_selectivity(self, 
                                     si_range=oplots.SI_RANGE,
                                     n_hist_bins=oplots.N_HIST_BINS,
                                     color=oplots.STIM_COLOR,
                                     p_value_max=oplots.P_VALUE_MAX,
                                     peak_dff_min=oplots.PEAK_DFF_MIN):

        # responsive cells 
        vis_cells = (self.peak.ptest_sg < p_value_max) & (self.peak.peak_dff_sg > peak_dff_min)

        # orientation selective cells
        osi_cells = vis_cells & (self.peak.osi_sg > si_range[0]) & (self.peak.osi_sg < si_range[1])

        peak_osi = self.peak.ix[osi_cells]
        osis = peak_osi.osi_sg.values

        oplots.plot_selectivity_cumulative_histogram(osis, 
                                                     "orientation selectivity index",
                                                     si_range=si_range,
                                                     n_hist_bins=n_hist_bins,
                                                     color=color)

    def plot_preferred_orientation(self,
                                   include_labels=False,
                                   si_range=oplots.SI_RANGE,
                                   color=oplots.STIM_COLOR,
                                   p_value_max=oplots.P_VALUE_MAX,
                                   peak_dff_min=oplots.PEAK_DFF_MIN):

        vis_cells = (self.peak.ptest_sg < p_value_max) & (self.peak.peak_dff_sg > peak_dff_min)    
        pref_oris = self.peak.ix[vis_cells].ori_sg.values
        pref_oris = [ self.orivals[pref_ori] for pref_ori in pref_oris ]

        angles, counts = np.unique(pref_oris, return_counts=True)

        oplots.plot_radial_histogram(angles, 
                                     counts,
                                     include_labels=include_labels,
                                     all_angles=self.orivals,
                                     direction=-1,
                                     offset=180.0,
                                     color=color)

        max_count = max(counts)

        center_x = 0.0
        center_y = 0.5 * max_count

        # dimensions to get plot to fit 
        h = 1.6 * max_count
        w = 2.4 * max_count

        plt.gca().set(xlim=(center_x - w*0.5, center_x + w*0.5),
                      ylim = (center_y - h*0.5, center_y + h*0.5),
                      aspect=1.0)

    def plot_preferred_spatial_frequency(self, 
                                         si_range=oplots.SI_RANGE,
                                         color=oplots.STIM_COLOR,
                                         p_value_max=oplots.P_VALUE_MAX,
                                         peak_dff_min=oplots.PEAK_DFF_MIN):

        vis_cells = (self.peak.ptest_sg < p_value_max) & (self.peak.peak_dff_sg > peak_dff_min)    
        pref_sfs = self.peak.ix[vis_cells].sf_sg.values

        oplots.plot_condition_histogram(pref_sfs, 
                                        self.sfvals[1:],
                                        color=color)

        plt.xlabel("spatial frequency (cycles/deg)")
        plt.ylabel("number of cells")

    def open_fan_plot(self, cell_specimen_id, include_labels=False):
        cell_id = self.peak_row_from_csid(self.peak, cell_specimen_id)

        df = self.mean_sweep_response[str(cell_id)]
        st = self.data_set.get_stimulus_table('static_gratings')
        mask = st.dropna(subset=['orientation']).index

        data = df.values

        cmin = self.response[0,0,0,cell_id,0]
        cmax = data.mean() + data.std()*3

        fp = cplots.FanPlotter.for_static_gratings()
        fp.plot(r_data=st.spatial_frequency.ix[mask].values,
                angle_data=st.orientation.ix[mask].values,
                group_data=st.phase.ix[mask].values,
                data=df.ix[mask].values,
                clim=[cmin, cmax])
        fp.show_axes(closed=False)

        if include_labels:
            fp.show_r_labels()
            fp.show_angle_labels()

    @staticmethod
    def from_analysis_file(data_set, analysis_file):
        sg = StaticGratings(data_set)

        try:
            sg.populate_stimulus_table()

            sg._sweep_response = pd.read_hdf(analysis_file, "analysis/sweep_response_sg")
            sg._mean_sweep_response = pd.read_hdf(analysis_file, "analysis/mean_sweep_response_sg")
            sg._peak = pd.read_hdf(analysis_file, "analysis/peak")

            with h5py.File(analysis_file, "r") as f:
                sg._response = f["analysis/response_sg"].value
                sg._binned_dx_sp = f["analysis/binned_dx_sp"].value
                sg._binned_cells_sp = f["analysis/binned_cells_sp"].value
                sg._binned_dx_vis = f["analysis/binned_dx_vis"].value
                sg._binned_cells_vis = f["analysis/binned_cells_vis"].value
        except Exception as e:
            raise MissingStimulusException(e.args)

        return sg
