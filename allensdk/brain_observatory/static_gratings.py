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
from .brain_observatory_exceptions import BrainObservatoryAnalysisException


class StaticGratings(StimulusAnalysis):
    """ Perform tuning analysis specific to static gratings stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.static_gratings')

    def __init__(self, data_set, **kwargs):
        super(StaticGratings, self).__init__(data_set, **kwargs)

        self._stim_table = StimulusAnalysis._PRELOAD
        self._sweeplength = StimulusAnalysis._PRELOAD
        self._interlength = StimulusAnalysis._PRELOAD
        self._extralength = StimulusAnalysis._PRELOAD
        self._orivals = StimulusAnalysis._PRELOAD
        self._sfvals = StimulusAnalysis._PRELOAD
        self._phasevals = StimulusAnalysis._PRELOAD
        self._number_ori = StimulusAnalysis._PRELOAD
        self._number_sf = StimulusAnalysis._PRELOAD
        self._number_phase = StimulusAnalysis._PRELOAD

        self._sweep_response = StimulusAnalysis._PRELOAD
        self._mean_sweep_response = StimulusAnalysis._PRELOAD
        self._pval = StimulusAnalysis._PRELOAD

        self._response = StimulusAnalysis._PRELOAD
        self._peak = StimulusAnalysis._PRELOAD

    @property
    def stim_table(self):
        if self._stim_table is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._stim_table

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

    @property
    def orivals(self):
        if self._orivals is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._orivals

    @property
    def sfvals(self):
        if self._sfvals is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._sfvals

    @property
    def phasevals(self):
        if self._phasevals is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._phasevals

    @property
    def number_ori(self):
        if self._number_ori is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._number_ori

    @property
    def number_sf(self):
        if self._number_sf is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._number_sf

    @property
    def number_phase(self):
        if self._number_phase is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._number_phase

    @property
    def sweep_response(self):
        if self._sweep_response is StimulusAnalysis._PRELOAD:
            self._sweep_response, self._mean_sweep_response, self._pval = \
                self.get_sweep_response()

        return self._sweep_response

    @property
    def mean_sweep_response(self):
        if self._mean_sweep_response is StimulusAnalysis._PRELOAD:
            self._sweep_response, self._mean_sweep_response, self._pval = \
                self.get_sweep_response()

        return self._mean_sweep_response

    @property
    def pval(self):
        if self._pval is StimulusAnalysis._PRELOAD:
            self._sweep_response, self._mean_sweep_response, self._pval = \
                self.get_sweep_response()

        return self._pval

    @property
    def response(self):
        if self._response is StimulusAnalysis._PRELOAD:
            self._response = self.get_response()

        return self._response

    @property
    def peak(self):
        if self._peak is StimulusAnalysis._PRELOAD:
            self._peak = self.get_peak()

        return self._peak

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
