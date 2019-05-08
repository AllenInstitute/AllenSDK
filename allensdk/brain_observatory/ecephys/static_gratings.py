from .stimulus_analysis import StimulusAnalysis
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi
from scipy.optimize import curve_fit


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
            self._get_response_events()

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
            peak = pd.DataFrame(np.empty(self.numbercells, dtype=np.dtype(self.PEAK_COLS)))
            #print(peak)
            #exit()
            #peak_df = pd.DataFrame(index=range(self.numbercells), columns=self.peak_columns)
            #for pcol, ptype in self.PEAK_COLS:
            #    # pandas can have issues interpreting type and makes the column 'object' type, this should enforce
            #    # the correct data type for each column
            #    print(pcol, ptype)
            #    peak_df[pcol] = peak_df[pcol].astype(dtype=ptype)
            #print(peak_df['cell_specimen_id'].dtype)
            #print(peak_df['num_pref_trials_sg'].dtype)
            #print(peak_df.dtypes)
            #exit()

            #dtypes = (np.uint64, np.float64, np.float64, np.float64, np.int64, np.bool, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64)
            #print(self.peak_dtypes)
            #peak = pd.DataFrame(index=range(self.numbercells),
            #                    columns=self.peak_columns,
            #                    dtype=np.dtype(self.PEAK_COLS),
            #                    # dtype=np.dtype([dtypes])
            #                    )
            #                    # dtype=self.peak_dtypes)
            #peak = pd.DataFrame(
            #    columns=('cell_specimen_id', 'pref_ori_sg', 'pref_sf_sg', 'pref_phase_sg', 'num_pref_trials_sg',
            #             'responsive_sg', 'g_osi_sg', 'sfdi_sg', 'reliability_sg', 'lifetime_sparseness_sg',
            #             'fit_sf_sg', 'fit_sf_ind_sg',
            #             'sf_low_cutoff_sg', 'sf_high_cutoff_sg', 'run_pval_sg', 'run_mod_sg', 'run_resp_sg',
            #             'stat_resp_sg'), index=range(self.numbercells))

            # set values to null by default
            peak.fit_sf_sg = np.nan
            peak.fit_sf_ind_sg = np.nan
            peak.sf_low_cutoff_sg = np.nan
            peak.sf_high_cutoff_sg = np.nan

            peak['lifetime_sparseness_sg'] = self._get_lifetime_sparseness()
            peak['cell_specimen_id'] = list(self.spikes.keys())
            for nc, v in enumerate(self.spikes.keys()):
                pref_ori = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())[0][0]
                pref_sf = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())[1][0]
                pref_phase = np.where(self.response_events[:, 1:, :, nc, 0] == self.response_events[:, 1:, :, nc, 0].max())[2][0]
                peak.pref_ori_sg.iloc[nc] = self.orivals[pref_ori]
                peak.pref_sf_sg.iloc[nc] = self.sfvals[pref_sf]
                peak.pref_phase_sg.iloc[nc] = self.phasevals[pref_phase]
                # print(int(self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2]))
                peak.num_pref_trials_sg.iloc[nc] = int(self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2])
                # print(peak['num_pref_trials_sg'].dtype)
                if self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2] > 11:
                    peak.responsive_sg.iloc[nc] = True
                else:
                    peak.responsive_sg.iloc[nc] = False

                peak.g_osi_sg.iloc[nc] = self._get_osi(pref_sf, pref_phase, nc)
                peak.reliability_sg.iloc[nc] = self._get_reliability(pref_ori, pref_sf, pref_phase, v)
                peak.sfdi_sg.iloc[nc] = self._get_sfdi(pref_ori, pref_phase, nc)
                peak.run_pval_sg.iloc[nc], peak.run_mod_sg.iloc[nc], peak.run_resp_sg.iloc[nc], peak.stat_resp_sg.iloc[
                    nc] = self._get_running_modulation(pref_ori, pref_sf, pref_phase, v)
                # SF fit only for responsive cells
                if self.response_events[pref_ori, pref_sf + 1, pref_phase, nc, 2] > 11:
                    peak.fit_sf_ind_sg.iloc[nc], peak.fit_sf_sg.iloc[nc], peak.sf_low_cutoff_sg.iloc[nc], \
                    peak.sf_high_cutoff_sg.iloc[nc] = self._fit_sf_tuning(pref_ori, pref_sf, pref_phase, nc)
                #else:
                #    peak.fit_sf_sg.iloc[nc] = np.nan


            # TEMPORARY, changing the above to use vectorization will fix the issue
            #peak['pref_sf_sg'] = peak['pref_sf_sg'].astype(np.float64)
            #print(peak['pref_sf_sg'].dtype)
            self._peak = peak

        #print(peak['num_pref_trials_sg'])
        #print(self.sfvals.dtype)
        #exit()
        # print(peak)

        return self._peak

    def _get_lifetime_sparseness(self):
        response = self.response_events[:, 1:, :, :, 0].reshape(120, self.numbercells)
        #print(response)
        #print(((1 - (1 / 120.) * ((np.power(response.sum(axis=0), 2)) / (np.power(response, 2).sum(axis=0)))) / (
        #            1 - (1 / 120.))))
        # exit()
        return ((1 - (1 / 120.) * ((np.power(response.sum(axis=0), 2)) / (np.power(response, 2).sum(axis=0)))) / (
                    1 - (1 / 120.)))

    def _get_response_events(self):
        # TODO: Check that SF=0.0 isn't in the data, it is a special condition coded into responses table
        response_events = np.empty((6, 6, 4, self.numbercells, 3))  # change to num_ori, num_sf+1, num_phase
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

        self._response_events = response_events
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
        tuning = self.response_events[:, pref_sf + 1, pref_phase, nc, 0]
        CV_top_os = np.empty(6, dtype=np.complex128)
        for i in range(6):
            CV_top_os[i] = (tuning[i] * np.exp(1j * 2 * orivals_rad[i]))
        return np.abs(CV_top_os.sum()) / tuning.sum()

    def _get_reliability(self, pref_ori, pref_sf, pref_phase, v):
        """computes trial-to-trial reliability of cell at its preferred condition

        :param pref_ori:
        :param pref_sf:
        :param pref_phase:
        :param v:
        :return:
        """
        subset = self.sweep_events[(self.stim_table['SF']==self.sfvals[pref_sf])
                                     &(self.stim_table['Ori']==self.orivals[pref_ori])
                                     &(self.stim_table['Phase']==self.phasevals[pref_phase])]

        subset += 1.
        corr_matrix = np.empty((len(subset),len(subset)))
        for i in range(len(subset)):
            fri = get_fr(subset[v].iloc[i])
            for j in range(len(subset)):
                frj = get_fr(subset[v].iloc[j])
                r,p = st.pearsonr(fri[30:40], frj[30:40])
                corr_matrix[i,j] = r

        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0],inds[1]]
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
        trials = self.mean_sweep_events[(self.stim_table['Ori'] == self.orivals[pref_ori]) & (
                    self.stim_table['Phase'] == self.phasevals[pref_phase])][v].values
        SSE_part = np.sqrt(np.sum((trials - trials.mean()) ** 2) / (len(trials) - 5))
        return (np.ptp(sf_tuning)) / (np.ptp(sf_tuning) + 2 * SSE_part)

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

    def _fit_sf_tuning(self, pref_ori, pref_sf, pref_phase, nc):
        """performs gaussian or exponential fit on the spatial frequency tuning curve at preferred orientation/phase.

        :param pref_ori:
        :param pref_sf:
        :param pref_phase:
        :param nc:
        :return: index for the preferred sf from the curve fit prefered sf from the curve fit low cutoff sf from the
        curve fit high cutoff sf from the curve fit
        """
        sf_tuning = self.response_events[pref_ori,1:,pref_phase,nc,0]
        fit_sf_ind = np.NaN
        fit_sf = np.NaN
        sf_low_cutoff = np.NaN
        sf_high_cutoff = np.NaN
        # print(pref_sf)
        if pref_sf in range(1,4):
            try:
                popt, pcov = curve_fit(gauss_function, range(5), sf_tuning, p0=[np.amax(sf_tuning), pref_sf, 1.], maxfev=2000)
                sf_prediction = gauss_function(np.arange(0., 4.1, 0.1), *popt)
                fit_sf_ind = popt[1]
                fit_sf = 0.02*np.power(2,popt[1])
                low_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[:sf_prediction.argmax()].argmin()
                high_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[sf_prediction.argmax():].argmin()+sf_prediction.argmax()
                if low_cut_ind>0:
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    sf_low_cutoff = 0.02*np.power(2, low_cutoff)
                elif high_cut_ind<49:
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    sf_high_cutoff = 0.02*np.power(2, high_cutoff)
            except:
                pass
        else:
            fit_sf_ind = pref_sf
            fit_sf = self.sfvals[pref_sf]
            try:
                popt, pcov = curve_fit(exp_function, range(5), sf_tuning, p0=[np.amax(sf_tuning), 2., np.amin(sf_tuning)], maxfev=2000)
                sf_prediction = exp_function(np.arange(0., 4.1, 0.1), *popt)
                if pref_sf==0:
                    high_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[sf_prediction.argmax():].argmin()+sf_prediction.argmax()
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    sf_high_cutoff = 0.02*np.power(2, high_cutoff)
                else:
                    low_cut_ind = np.abs(sf_prediction-(sf_prediction.max()/2.))[:sf_prediction.argmax()].argmin()
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    sf_low_cutoff = 0.02*np.power(2, low_cutoff)
            except:
                pass
        # print(fit_sf)
        return fit_sf_ind, fit_sf, sf_low_cutoff, sf_high_cutoff


def do_sweep_mean_shifted(x):
    return len(x[(x > 0.066) & (x < 0.316)])/0.25


def get_fr(spikes, num_timestep_second=30, filter_width=0.1):
    spikes = spikes.astype(float)
    spike_train = np.zeros((int(3.1*num_timestep_second))) #hardcoded 3 second sweep length
    spike_train[(spikes*num_timestep_second).astype(int)]=1
    filter_width = int(filter_width*num_timestep_second)
    fr = ndi.gaussian_filter(spike_train, filter_width)
    return fr


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c
