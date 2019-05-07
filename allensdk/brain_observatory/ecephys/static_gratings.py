from .stimulus_analysis import StimulusAnalysis
import numpy as np
import pandas as pd


class StaticGratings(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(StaticGratings, self).__init__(ecephys_session, **kwargs)
        self._sweep_p_values = None
        self._response_events = None
        self._response_trials = None
        self._peak = None

    @property
    def mean_sweep_events(self):
        if self._mean_sweep_events is None:
            # TODO: Should dtype for matrix be uint?
            self._mean_sweep_events = self.sweep_events.applymap(do_sweep_mean_shifted)

        return self._mean_sweep_events

    @property
    def sweep_p_values(self):
        if self._sweep_p_values is None:
            # TODO: Need to improve
            shuffled_mean = np.empty((self.numbercells, 10000))
            idx = np.random.choice(np.arange(self.stim_table_spontaneous['start_time'].iloc[0],
                                             self.stim_table_spontaneous['stop_time'].iloc[0],
                                             0.0001), 10000)  # TODO: what step size for np.arange?
            for shuf in range(10000):
                for i, v in enumerate(self.spikes.keys()):
                    spikes = self.spikes[v]
                    shuffled_mean[i, shuf] = len(spikes[(spikes > idx[shuf]) & (spikes < (idx[shuf] + 0.33))])

            sweep_p_values = pd.DataFrame(index=self.stim_table.index.values, columns=self.sweep_events.columns)
            for i, v in enumerate(self.spikes.keys()):
                subset = self.mean_sweep_events[v].values
                null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(subset), 1))
                actual_is_less = subset.reshape(len(subset), 1) <= null_dist_mat
                p_values = np.mean(actual_is_less, axis=1)
                sweep_p_values[v] = p_values

            self._sweep_p_values = sweep_p_values

        return self._sweep_p_values

    @property
    def response_events(self):
        if self._response_events is None:
            self._get_response_events()

        return self._response_events

    @property
    def response_trials(self):
        if self._response_trials is None:
            self._get_response_events()

        return self._response_trials

    @property
    def peak(self):
        if self._peak is None:
            peak = pd.DataFrame(
                columns=('cell_specimen_id', 'pref_ori_sg', 'pref_sf_sg', 'pref_phase_sg', 'num_pref_trials_sg',
                         'responsive_sg', 'g_osi_sg', 'sfdi_sg', 'reliability_sg', 'lifetime_sparseness_sg',
                         'fit_sf_sg', 'fit_sf_ind_sg',
                         'sf_low_cutoff_sg', 'sf_high_cutoff_sg', 'run_pval_sg', 'run_mod_sg', 'run_resp_sg',
                         'stat_resp_sg'), index=range(self.numbercells))

            peak['lifetime_sparseness_dg'] = self._get_lifetime_sparseness()
            peak['cell_specimen_id'] = list(self.spikes.keys())
            for nc, v in enumerate(self.spikes.keys()):
                #print(v)
                #print(self.response_events[:, 1:, :, nc, 0])
                #print(np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max()))
                pref_ori = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())[0][0]
                pref_sf = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())[1][0]
                pref_phase = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())[2][0]
                peak.pref_ori_sg.iloc[nc] = self.orivals[pref_ori]
                peak.pref_sf_sg.iloc[nc] = self.sfvals[pref_sf]
                peak.pref_phase_sg.iloc[nc] = self.phasevals[pref_phase]
                peak.num_pref_trials_sg.iloc[nc] = self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2]
                if self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2] > 11:
                    peak.responsive_sg.iloc[nc] = True
                else:
                    peak.responsive_sg.iloc[nc] = False

                peak.g_osi_sg.iloc[nc] = self.get_osi(pref_sf, pref_phase, nc)
                peak.reliability_sg.iloc[nc] = self.get_reliability(pref_ori, pref_sf, pref_phase, v)
                peak.sfdi_sg.iloc[nc] = self.get_sfdi(pref_ori, pref_phase, nc)
                peak.run_pval_sg.iloc[nc], peak.run_mod_sg.iloc[nc], peak.run_resp_sg.iloc[nc], peak.stat_resp_sg.iloc[
                    nc] = self.get_running_modulation(pref_ori, pref_sf, pref_phase, v)
                # SF fit only for responsive cells
                if self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2] > 11:
                    peak.fit_sf_ind_sg.iloc[nc], peak.fit_sf_sg.iloc[nc], peak.sf_low_cutoff_sg.iloc[nc], \
                    peak.sf_high_cutoff_sg.iloc[nc] = self.fit_sf_tuning(pref_ori, pref_sf, pref_phase, nc)
            self._peak = peak

        return self._peak

    def _get_lifetime_sparseness(self):
        response = self.response_events[:, 1:, :, :, 0].reshape(120, self.numbercells)
        return ((1 - (1 / 120.) * ((np.power(response.sum(axis=0), 2)) / (np.power(response, 2).sum(axis=0)))) / (
                    1 - (1 / 120.)))


    def _get_response_events(self):
        response_events = np.empty((6, 6, 4, self.numbercells, 3))
        response_trials = np.empty((6, 6, 4, self.numbercells, 50))
        response_trials[:] = np.nan
        for oi, ori in enumerate(self.orivals):
            for si, sf in enumerate(self.sfvals):
                # print(sf)
                for phi, phase in enumerate(self.phasevals):
                    subset = self.mean_sweep_events[
                        (self.stim_table['Ori'] == ori) & (self.stim_table['SF'] == sf) & (
                                    self.stim_table['Phase'] == phase)]
                    subset_p = self.sweep_p_values[
                        (self.stim_table['Ori'] == ori) & (self.stim_table['SF'] == sf) & (
                                    self.stim_table['Phase'] == phase)]
                    response_events[oi, si + 1, phi, :, 0] = subset.mean(axis=0)
                    response_events[oi, si + 1, phi, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
                    response_events[oi, si + 1, phi, :, 2] = subset_p[subset_p < 0.05].count().values
                    response_trials[oi, si + 1, phi, :, :subset.shape[0]] = subset.values.T

        # print(self.stim_table['SF'])
        subset = self.mean_sweep_events[np.isnan(self.stim_table['Ori'])]
        subset_p = self.sweep_p_values[np.isnan(self.stim_table['Ori'])]
        response_events[0, 0, 0, :, 0] = subset.mean(axis=0)
        response_events[0, 0, 0, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
        response_events[0, 0, 0, :, 2] = subset_p[subset_p < 0.05].count().values

        #print(response_events.shape)
        #print(response_events)
        #exit()

        self._response_events = response_events
        self._response_trials = response_trials


def do_sweep_mean_shifted(x):
    return len(x[(x > 0.066) & (x < 0.316)])/0.25
