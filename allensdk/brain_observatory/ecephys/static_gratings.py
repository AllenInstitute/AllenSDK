from .stimulus_analysis import StimulusAnalysis
import numpy as np
from six import string_types
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi
from scipy.optimize import curve_fit


class StaticGratings(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(StaticGratings, self).__init__(ecephys_session, **kwargs)
        self._orivals = None
        self._number_ori = None
        self._sfvals = None
        self._number_sf = None
        self._phasevals = None
        self._number_phase = None
        # self._sweep_p_values = None
        self._response_events = None
        self._response_trials = None
        # self._peak = None

    @property
    def stim_table(self):
        # Stimulus table is already in EcephysSession object, just need to subselect 'static_gratings' presentations.
        if self._stim_table is None:
            # TODO: Give warning if no static_gratings stimulus
            if self._stimulus_names is None:
                # Older versions of NWB files the stimulus name is in the form stimulus_gratings_N, so if
                # self._stimulus_names is not explicity specified try to figure out stimulus
                stims_table = self.ecephys_session.stimulus_presentations
                stim_names = [s for s in stims_table['stimulus_name'].unique()
                              if s.lower().startswith('static_gratings')]

                self._stim_table = stims_table[stims_table['stimulus_name'].isin(stim_names)]

            else:
                self._stimulus_names = [self._stimulus_names] if isinstance(self._stimulus_names, string_types) \
                    else self._stimulus_names
                self._stim_table = self.ecephys_session.get_presentations_for_stimulus(self._stimulus_names)

        return self._stim_table

    @property
    def orivals(self):
        if self._orivals is None:
            self._get_stim_table_stats()

        return self._orivals

    @property
    def number_ori(self):
        if self._number_ori is None:
            self._get_stim_table_stats()

        return self._number_ori

    @property
    def sfvals(self):
        if self._sfvals is None:
            self._get_stim_table_stats()

        return self._sfvals

    @property
    def number_sf(self):
        if self._number_sf is None:
            self._get_stim_table_stats()

        return self._number_sf

    @property
    def phasevals(self):
        if self._phasevals is None:
            self._get_stim_table_stats()

        return self._phasevals

    @property
    def number_phase(self):
        if self._number_phase is None:
            self._get_stim_table_stats()

        return self._number_phase

    @property
    def mean_sweep_events(self):
        if self._mean_sweep_events is None:
            # TODO: Should dtype for matrix be uint?
            self._mean_sweep_events = self.sweep_events.applymap(do_sweep_mean_shifted)

        return self._mean_sweep_events

    @property
    def sweep_p_values(self):
        if self._sweep_p_values is None:
            # TODO: Code is currently a speed bottle-neck and could probably be improved.
            # Recreate the mean-sweep-table but using randomly selected 'spontaneuous' stimuli.
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
            self._get_response_trials()

        return self._response_trials

    PEAK_COLS = [('cell_specimen_id', np.uint64), ('pref_ori_sg', np.float64), ('pref_sf_sg', np.float64),
                 ('pref_phase_sg', np.float64), ('num_pref_trials_sg', np.uint64), ('responsive_sg', np.bool),
                 ('g_osi_sg', np.float64), ('sfdi_sg', np.float64), ('reliability_sg', np.float64),
                 ('lifetime_sparseness_sg', np.float64), ('fit_sf_sg', np.float64), ('fit_sf_ind_sg', np.float64),
                 ('sf_low_cutoff_sg', np.float64), ('sf_high_cutoff_sg', np.float64), ('run_pval_sg', np.float64),
                 ('run_mod_sg', np.float64), ('run_resp_sg', np.float64), ('stat_resp_sg', np.float64)]

    @property
    def peak_columns(self):
        return [c[0] for c in self.PEAK_COLS]

    @property
    def peak_dtypes(self):
        return [c[1] for c in self.PEAK_COLS]

    @property
    def peak(self):
        if self._peak is None:
            # pandas can have issues interpreting type and makes the column 'object' type, this should enforce the
            # correct data type for each column
            peak_df = pd.DataFrame(np.empty(self.numbercells, dtype=np.dtype(self.PEAK_COLS)),
                                   index=range(self.numbercells))

            # set values to null by default
            peak_df['fit_sf_sg'] = np.nan
            peak_df['fit_sf_ind_sg'] = np.nan
            peak_df['sf_low_cutoff_sg'] = np.nan
            peak_df['sf_high_cutoff_sg'] = np.nan

            peak_df['cell_specimen_id'] = list(self.spikes.keys())
            peak_df['lifetime_sparseness_sg'] = self._get_lifetime_sparseness()

            for nc, unit_id in enumerate(self.spikes.keys()):
                peaks = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())
                pref_ori = peaks[0][0]
                pref_sf = peaks[1][0]
                pref_phase = peaks[2][0]

                peak_df.loc[nc, 'pref_ori_sg'] = self.orivals[pref_ori]
                peak_df.loc[nc, 'pref_sf_sg'] = self.sfvals[pref_sf]
                peak_df.loc[nc, 'pref_phase_sg'] = self.phasevals[pref_phase]

                peak_df.loc[nc, 'num_pref_trials_sg'] = int(self.response_events[pref_ori, pref_sf+1, pref_phase, nc, 2])
                peak_df.loc[nc, 'responsive_sg'] = self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2] > 11

                stim_table_mask = (self.stim_table['SF'] == self.sfvals[pref_sf]) & \
                                  (self.stim_table['Ori'] == self.orivals[pref_ori]) & \
                                  (self.stim_table['Phase'] == self.phasevals[pref_phase])
                peak_df.loc[nc, 'g_osi_sg'] = self._get_osi(pref_sf, pref_phase, nc)

                peak_df.loc[nc, 'reliability_sg'] = self._get_reliability(unit_id, stim_table_mask)
                peak_df.loc[nc, 'sfdi_sg'] = self._get_sfdi(pref_ori, pref_phase, nc)
                peak_df.loc[nc, ['run_pval_sg', 'run_mod_sg', 'run_resp_sg', 'stat_resp_sg']] = \
                    self._get_running_modulation(unit_id, stim_table_mask)

                if self.response_events[pref_ori, pref_sf+1, pref_phase, nc, 2] > 11:
                    peak_df.loc[nc, ['fit_sf_ind_sg', 'fit_sf_sg', 'sf_low_cutoff_sg', 'sf_high_cutoff_sg']] = \
                        self._fit_sf_tuning(pref_ori, pref_sf, pref_phase, nc)

            self._peak = peak_df

        return self._peak

    def _get_stim_table_stats(self):
        sg_stim_table = self.stim_table
        self._orivals = np.sort(sg_stim_table['Ori'].dropna().unique())
        self._number_ori = len(self._orivals)

        self._sfvals = np.sort(sg_stim_table['SF'].dropna().unique())
        # TODO: Check that SF=0.0 isn't in the data, it is a special condition coded into responses table
        self._number_sf = len(self._sfvals)

        self._phasevals = np.sort(sg_stim_table['Phase'].dropna().unique())
        self._number_phase = len(self._phasevals)

    def _get_lifetime_sparseness(self):
        response = self.response_events[:, 1:, :, :, 0].reshape(120, self.numbercells)
        return ((1 - (1 / 120.) * ((np.power(response.sum(axis=0), 2)) / (np.power(response, 2).sum(axis=0)))) / (
                    1 - (1 / 120.)))

    def _get_response_events(self):
        # for each cell, find all trials with the same orientation/spatial_freq/phase/cell combo; get averaged num
        # of spikes, the standard err, and a count of all statistically significant trials.
        response_events = np.empty((self.number_ori, self.number_sf+1, self.number_phase, self.numbercells, 3))
        for oi, ori in enumerate(self.orivals):
            ori_mask = self.stim_table['Ori'] == ori
            for si, sf in enumerate(self.sfvals):
                sf_mask = self.stim_table['SF'] == sf
                for phi, phase in enumerate(self.phasevals):
                    mask = ori_mask & sf_mask & (self.stim_table['Phase'] == phase)
                    subset = self.mean_sweep_events[mask]
                    subset_p = self.sweep_p_values[mask]

                    response_events[oi, si+1, phi, :, 0] = subset.mean(axis=0)
                    response_events[oi, si+1, phi, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
                    response_events[oi, si+1, phi, :, 2] = subset_p[subset_p < 0.05].count().values

        # A special case for a blank (or invalid?) stimulus
        subset = self.mean_sweep_events[np.isnan(self.stim_table['Ori'])]
        subset_p = self.sweep_p_values[np.isnan(self.stim_table['Ori'])]
        response_events[0, 0, 0, :, 0] = subset.mean(axis=0)
        response_events[0, 0, 0, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
        response_events[0, 0, 0, :, 2] = subset_p[subset_p < 0.05].count().values

        self._response_events = response_events

    def _get_response_trials(self):
        # Similar to special response_events, but instead of storing mean_sweep statistics stores the actual values.
        # TODO: Assumes that there are an equal number of trials for every ori/sf/phase combo. Will fail if not
        # TODO: This dataset is not being used by other part of class, and there is no analog in ophys. Should
        #    consider removing this altogether?
        n_stims = len(self.stim_table)
        n_features = self.number_ori * self.number_sf * self.number_phase
        response_trials = np.empty((self.number_ori, self.number_sf+1, self.number_phase, self.numbercells,
                                    n_stims//n_features))
        response_trials[0, 0, 0, :, :] = np.nan
        for oi, ori in enumerate(self.orivals):
            ori_mask = self.stim_table['Ori'] == ori
            for si, sf in enumerate(self.sfvals):
                sf_mask = self.stim_table['SF'] == sf
                for phi, phase in enumerate(self.phasevals):
                    mask = ori_mask & sf_mask & (self.stim_table['Phase'] == phase)
                    subset = self.mean_sweep_events[mask]
                    response_trials[oi, si+1, phi, :, :subset.shape[0]] = subset.values.T

        self._response_trials = response_trials

    def _get_osi(self, pref_sf, pref_phase, nc):
        """computes orientation selectivity (cv) for cell

        :param self:
        :param pref_sf:
        :param pref_phase:
        :param nc:
        :return:
        """
        orivals_rad = np.deg2rad(self.orivals)
        tuning = self.response_events[:, pref_sf+1, pref_phase, nc, 0]
        cv_top_os = np.empty(self.number_ori, dtype=np.complex128)
        for i in range(self.number_ori):
            cv_top_os[i] = (tuning[i] * np.exp(1j*2*orivals_rad[i]))

        return np.abs(cv_top_os.sum()) / tuning.sum()

    def _get_reliability(self, specimen_id, st_mask):
        """computes trial-to-trial reliability of cell at its preferred condition

        :param specimen_id:
        :param st_mask:
        :return:
        """
        subset = self.sweep_events[st_mask][specimen_id].values
        # print(subset)
        #subset = self.sweep_events[(self.stim_table['SF']==self.sfvals[pref_sf]) &
        #                           (self.stim_table['Ori']==self.orivals[pref_ori]) &
        #                           (self.stim_table['Phase']==self.phasevals[pref_phase])]

        subset += 1.0
        corr_matrix = np.empty((len(subset), len(subset)))
        for i in range(len(subset)):
            fri = get_fr(subset[i])
            # fri = get_fr(subset[v].iloc[i])
            for j in range(len(subset)):
                # frj = get_fr(subset[v].iloc[j])
                frj = get_fr(subset[j])
                r, p = st.pearsonr(fri[30:40], frj[30:40])
                corr_matrix[i, j] = r

        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0], inds[1]]
        return np.nanmean(upper)

    def _get_sfdi(self, pref_ori, pref_phase, nc):
        """computes spatial frequency discrimination index for cell

        :param pref_ori:
        :param pref_phase:
        :param nc:
        :return: sf discrimination index
        """
        v = list(self.spikes.keys())[nc]
        sf_tuning = self.response_events[pref_ori, 1:, pref_phase, nc, 0]
        trials = self.mean_sweep_events[(self.stim_table['Ori'] == self.orivals[pref_ori]) &
                                        (self.stim_table['Phase'] == self.phasevals[pref_phase])][v].values
        sse_part = np.sqrt(np.sum((trials - trials.mean())**2) / (len(trials)-5))

        return (np.ptp(sf_tuning)) / (np.ptp(sf_tuning) + 2 * sse_part)

    def _get_running_modulation(self, unit_id, st_mask):
        """computes running modulation of cell at its preferred condition provided there are at least 2 trials for both
        stationary and running conditions

        :param unit_id:
        :param st_mask:
        :return: p_value of running modulation, running modulation metric, mean response to preferred condition when
        running mean response to preferred condition when stationary
        """
        subset = self.mean_sweep_events[st_mask]
        speed_subset = self.running_speed[st_mask]

        subset_run = subset[speed_subset.running_speed >= 1][unit_id]
        subset_stat = subset[speed_subset.running_speed < 1][unit_id]
        if np.logical_and(len(subset_run) > 1, len(subset_stat) > 1):
            run = subset_run.mean()
            stat = subset_stat.mean()
            if run > stat:
                run_mod = (run - stat) / run
            elif stat > run:
                run_mod = -1 * (stat - run) / stat
            else:
                run_mod = 0
            (_, p) = st.ttest_ind(subset_run, subset_stat, equal_var=False)
            return p, run_mod, run, stat
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN

    '''
    def _get_running_modulation(self, pref_ori, pref_sf, pref_phase, v):
        """computes running modulation of cell at its preferred condition provided there are at least 2 trials for both
        stationary and running conditions

        :param pref_ori:
        :param pref_sf:
        :param pref_phase:
        :param v:
        :return: p_value of running modulation, running modulation metric, mean response to preferred condition when
        running mean response to preferred condition when stationary
        """
        subset = self.mean_sweep_events[(self.stim_table['SF'] == self.sfvals[pref_sf])
                                        & (self.stim_table['Ori'] == self.orivals[pref_ori])
                                        & (self.stim_table['Phase'] == self.phasevals[pref_phase])]
        speed_subset = self.running_speed[(self.stim_table['SF'] == self.sfvals[pref_sf])
                                          & (self.stim_table['Ori'] == self.orivals[pref_ori])
                                          & (self.stim_table['Phase'] == self.phasevals[pref_phase])]

        subset_run = subset[speed_subset.running_speed >= 1]
        subset_stat = subset[speed_subset.running_speed < 1]
        # print(len(subset_run), len(subset_stat))
        if np.logical_and(len(subset_run) > 1, len(subset_stat) > 1):
            # print('HERE')
            run = subset_run[v].mean()
            stat = subset_stat[v].mean()
            if run > stat:
                run_mod = (run - stat) / run
            elif stat > run:
                run_mod = -1 * (stat - run) / stat
            else:
                run_mod = 0
            (_, p) = st.ttest_ind(subset_run[v], subset_stat[v], equal_var=False)
            return p, run_mod, run, stat
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN
    '''

    def _fit_sf_tuning(self, pref_ori, pref_sf, pref_phase, nc):
        """performs gaussian or exponential fit on the spatial frequency tuning curve at preferred orientation/phase.

        :param pref_ori:
        :param pref_sf:
        :param pref_phase:
        :param nc:
        :return: index for the preferred sf from the curve fit prefered sf from the curve fit low cutoff sf from the
        curve fit high cutoff sf from the curve fit
        """
        sf_tuning = self.response_events[pref_ori, 1:, pref_phase, nc, 0]
        fit_sf_ind = np.NaN
        fit_sf = np.NaN
        sf_low_cutoff = np.NaN
        sf_high_cutoff = np.NaN
        print(pref_sf)
        if pref_sf in range(1, 4):  # TODO: Is this correct?
            try:
                popt, pcov = curve_fit(gauss_function, range(5), sf_tuning, p0=[np.amax(sf_tuning), pref_sf, 1.], maxfev=2000)
                sf_prediction = gauss_function(np.arange(0., 4.1, 0.1), *popt)
                fit_sf_ind = popt[1]
                fit_sf = 0.02*np.power(2, popt[1])
                low_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[:sf_prediction.argmax()].argmin()
                high_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[sf_prediction.argmax():].argmin() + sf_prediction.argmax()
                if low_cut_ind > 0:
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    sf_low_cutoff = 0.02*np.power(2, low_cutoff)
                elif high_cut_ind < 49:
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    sf_high_cutoff = 0.02*np.power(2, high_cutoff)
            except Exception:
                pass
        else:
            fit_sf_ind = pref_sf
            fit_sf = self.sfvals[pref_sf]
            try:
                popt, pcov = curve_fit(exp_function, range(5), sf_tuning,
                                       p0=[np.amax(sf_tuning), 2., np.amin(sf_tuning)], maxfev=2000)
                sf_prediction = exp_function(np.arange(0., 4.1, 0.1), *popt)
                if pref_sf == 0:
                    high_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[sf_prediction.argmax():].argmin()+sf_prediction.argmax()
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    sf_high_cutoff = 0.02*np.power(2, high_cutoff)
                else:
                    low_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[:sf_prediction.argmax()].argmin()
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    sf_low_cutoff = 0.02*np.power(2, low_cutoff)
            except Exception:
                pass

        return fit_sf_ind, fit_sf, sf_low_cutoff, sf_high_cutoff


def do_sweep_mean_shifted(x):
    return len(x[(x > 0.066) & (x < 0.316)])/0.25


def get_fr(spikes, num_timestep_second=30, filter_width=0.1):
    spikes = spikes.astype(float)
    spike_train = np.zeros((int(3.1*num_timestep_second)))  # hardcoded 3 second sweep length
    spike_train[(spikes*num_timestep_second).astype(int)] = 1
    filter_width = int(filter_width*num_timestep_second)
    fr = ndi.gaussian_filter(spike_train, filter_width)
    return fr


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c
