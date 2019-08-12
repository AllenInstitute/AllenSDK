import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from allensdk.brain_observatory.ecephys.stimulus_analysis import StimulusAnalysis
from allensdk.brain_observatory.ecephys.stimulus_analysis import get_lifetime_sparseness, get_osi, get_dsi, \
    get_reliability, get_running_modulation


class DriftingGratings(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(DriftingGratings, self).__init__(ecephys_session, **kwargs)

        self._orivals = None
        self._number_ori = None
        self._tfvals = None
        self._number_tf = None
        self._response_events = None
        self._response_trials = None

        self._col_tf = 'TF'
        self._col_ori = 'Ori'

        # Used to determine responsivness metric and if their is enough activity to try to fit a cell's tf values.
        # TODO: Figure out how this value existed, possibly make it a user parameter?
        self._responsivness_threshold = kwargs.get('responsivness_threshold', 3)

        # Used to determine if a spontaneous repsonse if statistically significant
        self._response_events_p_val = kwargs.get('response_events_p_val', 0.05)

    PEAK_COLS = [('cell_specimen_id', np.uint64), ('pref_ori_dg', np.float64), ('pref_tf_dg', np.float64),
                 ('num_pref_trials_dg', np.uint64), ('responsive_dg', bool), ('g_osi_dg', np.float64),
                 ('dsi_dg', np.float64), ('tfdi_dg', np.float64), ('reliability_dg', np.float64),
                 ('lifetime_sparseness_dg', np.float64), ('fit_tf_dg', np.float64), ('fit_tf_ind_dg', np.float64),
                 ('tf_low_cutoff_dg', np.float64), ('tf_high_cutoff_dg', np.float64), ('run_pval_dg', np.float64),
                 ('run_resp_dg', np.float64), ('stat_resp_dg', np.float64), ('run_mod_dg', np.float64),
                 ('peak_blank_dg', np.float64), ('all_blank_dg', np.float64)]

    @property
    def peak_columns(self):
        return [c[0] for c in self.PEAK_COLS]

    @property
    def peak_dtypes(self):
        return [c[1] for c in self.PEAK_COLS]

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
    def tfvals(self):
        if self._tfvals is None:
            self._get_stim_table_stats()

        return self._tfvals

    @property
    def number_tf(self):
        if self._tfvals is None:
            self._get_stim_table_stats()

        return self._number_tf

    @property
    def stim_table(self):
        # Stimulus table is already in EcephysSession object, just need to subselect 'static_gratings' presentations.
        if self._stim_table is None:
            # TODO: Give warning if no stimulus
            if self._stimulus_names is None:
                # Older versions of NWB files the stimulus name is in the form stimulus_gratings_N, so if
                # self._stimulus_names is not explicity specified try to figure out stimulus
                stims_table = self.ecephys_session.stimulus_presentations
                stim_names = [s for s in stims_table['stimulus_name'].unique()
                              if s.lower().startswith('drifting_gratings')]

                self._stim_table = stims_table[stims_table['stimulus_name'].isin(stim_names)]

            else:
                self._stimulus_names = [self._stimulus_names] if isinstance(self._stimulus_names, string_types) \
                    else self._stimulus_names
                self._stim_table = self.ecephys_session.get_presentations_for_stimulus(self._stimulus_names)

        return self._stim_table

    @property
    def mean_sweep_events(self):
        if self._mean_sweep_events is None:
            # TODO: Should dtype for matrix be uint?
            self._mean_sweep_events = self.sweep_events.applymap(do_sweep_mean)

        return self._mean_sweep_events

    @property
    def sweep_p_values(self):
        if self._sweep_p_values is None:
            self._sweep_p_values = self.calc_sweep_p_values(offset=2.0)

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
            peak_df = pd.DataFrame(np.empty(self.numbercells, dtype=np.dtype(self.PEAK_COLS)),
                                   index=range(self.numbercells))

            peak_df['fit_tf_ind_dg'] = np.nan
            peak_df['fit_tf_dg'] = np.nan
            peak_df['tf_low_cutoff_dg'] = np.nan
            peak_df['tf_high_cutoff_dg'] = np.nan

            responses = self.response_events[:, 1:, :, 0].reshape(self.number_tf*self.number_ori, self.numbercells)
            peak_df['lifetime_sparseness_dg'] = get_lifetime_sparseness(responses)
            peak_df['cell_specimen_id'] = list(self.spikes.keys())

            for nc, unit_id in enumerate(self.spikes.keys()):
                peaks = np.where(self.response_events[:, 1:, nc, 0] == self.response_events[:, 1:, nc, 0].max())
                pref_ori = peaks[0][0]
                pref_tf = peaks[1][0]

                stim_table_mask = (self.stim_table[self._col_tf] == self.tfvals[pref_tf]) & \
                                  (self.stim_table[self._col_ori] == self.orivals[pref_ori])

                peak_df.loc[nc, 'pref_ori_dg'] = self.orivals[pref_ori]
                peak_df.loc[nc, 'pref_tf_dg'] = self.tfvals[pref_tf]
                peak_df.loc[nc, 'num_pref_trials_dg'] = self.response_events[pref_ori, pref_tf + 1, nc, 2]
                peak_df.loc[nc, 'responsive_dg'] = self.response_events[pref_ori, pref_tf + 1, nc, 2] > self._responsivness_threshold

                ori_tuning_responses = self.response_events[:, pref_tf+1, nc, 0]
                peak_df.loc[nc, 'g_osi_dg'] = get_osi(ori_tuning_responses, self.orivals)
                peak_df.loc[nc, 'dsi_dg'] = get_dsi(ori_tuning_responses, self.orivals)

                pref_sweeps = self.sweep_events[stim_table_mask][unit_id].values
                peak_df.loc[nc, 'reliability_dg'] = get_reliability(pref_sweeps, window_beg=30)

                tf_tuning_responses = self.response_events[pref_ori, 1:, nc, 0]
                trials = self.mean_sweep_events[(self.stim_table[self._col_ori] == self.orivals[pref_ori])][unit_id].values
                peak_df.loc[nc, 'tfdi_dg'] = get_tfdi(tf_tuning_responses, trials, self.number_tf)

                speed_subset = self.running_speed[stim_table_mask]
                subset = self.mean_sweep_events[stim_table_mask]
                mse_subset_run = subset[speed_subset['running_speed'] >= 1][unit_id].values
                mse_subset_stat = subset[speed_subset['running_speed'] < 1][unit_id].values
                peak_df.loc[nc, ['run_pval_dg', 'run_mod_dg', 'run_resp_dg', 'stat_resp_dg']] = \
                    get_running_modulation(mse_subset_run, mse_subset_stat)

                peak_responses = self.response_events[pref_ori, pref_tf+1, nc, 0]
                blank_responses = self.response_events[0, 0, nc, 0]
                all_responses = self.response_events[:, 1:, nc, 0]
                peak_df.loc[nc, ['peak_blank_dg', 'all_blank_dg']] = get_suppressed_contrast(peak_responses, all_responses,
                                                                                             blank_responses)

                if self.response_events[pref_ori, pref_tf+1, nc, 2] > self._responsivness_threshold:
                    peak_df.loc[nc, ['fit_tf_ind_dg', 'fit_tf_dg', 'tf_low_cutoff_dg', 'tf_high_cutoff_dg']] = \
                        get_fit_tf_tuning(tf_tuning_responses, tf_values=self.tfvals, pref_tf_index=pref_tf)

            self._peak = peak_df

        return self._peak

    def _get_response_events(self):
        response_events = np.empty((self.number_ori, self.number_tf+1, self.numbercells, 3))
        response_events[:] = np.NaN

        blank = self.mean_sweep_events[np.isnan(self.stim_table[self._col_ori])]
        response_trials = np.empty((self.number_ori, self.number_tf+1, self.numbercells, len(blank)))
        response_trials[:] = np.NaN

        response_events[0, 0, :, 0] = blank.mean(axis=0)
        response_events[0, 0, :, 1] = blank.std(axis=0) / np.sqrt(len(blank))
        blank_p = self.sweep_p_values[np.isnan(self.stim_table['Ori'])]
        response_events[0, 0, :, 2] = blank_p[blank_p < self._response_events_p_val].count().values
        response_trials[0, 0, :, :] = blank.values.T

        for oi, ori in enumerate(self.orivals):
            ori_mask = self.stim_table[self._col_ori] == ori
            for ti, tf in enumerate(self.tfvals):
                mask = ori_mask & (self.stim_table[self._col_tf] == tf)
                subset = self.mean_sweep_events[mask]
                subset_p = self.sweep_p_values[mask]
                response_events[oi, ti + 1, :, 0] = subset.mean(axis=0)
                response_events[oi, ti + 1, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
                response_events[oi, ti + 1, :, 2] = subset_p[subset_p < self._response_events_p_val].count().values
                response_trials[oi, ti + 1, :, :subset.shape[0]] = subset.values.T

        self._response_events = response_events
        self._response_trials = response_trials

    def _get_stim_table_stats(self):
        self._orivals = np.sort(self.stim_table[self._col_ori].dropna().unique())
        self._number_ori = len(self._orivals)

        self._tfvals = np.sort(self.stim_table[self._col_tf].dropna().unique())
        self._number_tf = len(self._tfvals)


def get_fit_tf_tuning(tf_tuning_responses, tf_values, pref_tf_index):
    """Performs gaussian or exponential fit on the temporal frequency tuning curve at preferred orientation.

    :param tf_tuning_responses: n array of len N, with each value the (averaged) response of a cell at a given temporal
        freq. stimulus.
    :param tf_values: An array of len N, with each value the spatial freq. of the stimulus (corresponding to
        tf_tuning_response).
    :param pref_tf_index: The pre-determined prefered temporal frequency (tf_values index) of the cell.
    :return: index for the preferred tf from the curve fit, prefered sf from the curve fit, low cutoff sf from the
        curve fit, high cutoff sf from the curve fit
    """
    fit_tf_ind = np.NaN
    fit_tf = np.NaN
    tf_low_cutoff = np.NaN
    tf_high_cutoff = np.NaN
    if pref_tf_index in range(1, len(tf_values)-1):
        try:
            popt, pcov = curve_fit(gauss_function, np.arange(len(tf_values)), tf_tuning_responses,
                                   p0=[np.amax(tf_tuning_responses), pref_tf_index, 1.], maxfev=2000)
            tf_prediction = gauss_function(np.arange(0., 4.1, 0.1), *popt)
            fit_tf_ind = popt[1]
            fit_tf = np.power(2, popt[1])
            low_cut_ind = np.abs(tf_prediction - (tf_prediction.max() / 2.))[:tf_prediction.argmax()].argmin()
            high_cut_ind = np.abs(tf_prediction - (tf_prediction.max() / 2.))[
                           tf_prediction.argmax():].argmin() + tf_prediction.argmax()
            if low_cut_ind > 0:
                low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                tf_low_cutoff = np.power(2, low_cutoff)
            elif high_cut_ind < 4:
                high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                tf_high_cutoff = np.power(2, high_cutoff)
        except Exception:
            pass
    else:
        fit_tf_ind = pref_tf_index
        fit_tf = tf_values[pref_tf_index]
        try:
            popt, pcov = curve_fit(exp_function, np.arange(len(tf_values)), tf_tuning_responses,
                                   p0=[np.amax(tf_tuning_responses), 2., np.amin(tf_tuning_responses)], maxfev=2000)
            tf_prediction = exp_function(np.arange(0., 4.1, 0.1), *popt)
            if pref_tf_index == 0:
                high_cut_ind = np.abs(tf_prediction - (tf_prediction.max() / 2.))[
                               tf_prediction.argmax():].argmin() + tf_prediction.argmax()
                high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                tf_high_cutoff = np.power(2, high_cutoff)
            else:
                low_cut_ind = np.abs(tf_prediction - (tf_prediction.max() / 2.))[
                              :tf_prediction.argmax()].argmin()
                low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                tf_low_cutoff = np.power(2, low_cutoff)
        except Exception:
            pass

    return fit_tf_ind, fit_tf, tf_low_cutoff, tf_high_cutoff


def get_tfdi(tf_tuning_responses, mean_sweeps_trials, bias=5):
    """Computes temporal frequency discrimination index (tfdi) for a given cell

    :param tf_tuning_responses: An array of len N, with each value the (averaged) response of a
        cell at a given temporal freq. stimulus.
    :param mean_sweeps_trials: The set of events (spikes) across all trials of varying
    :param bias:
    :return: The tfdi value (float)
    """
    # TODO: Should we merge this with StaticGratings.get_sfdi
    trial_mean = mean_sweeps_trials.mean()
    sse_part = np.sqrt(np.sum((mean_sweeps_trials - trial_mean)**2) / (len(mean_sweeps_trials) - bias))
    return (np.ptp(tf_tuning_responses)) / (np.ptp(tf_tuning_responses) + 2 * sse_part)


def get_suppressed_contrast(peak_response, all_responses, blank_response):
    """Computes two metrics to be used to identify cells that are suppressed by contrast

    :param peak_response: float, the mean cell response at the preferred grating
    :param all_responses: An array of all (non-blank) responses across all the gratings
    :param blank_response: float, the mean cell response for a blank input
    :return: A tuple of floats, the range between peak and blank responses, range between all and blank responses.
    """
    all_resp_mean = all_responses.mean()
    peak_blank = peak_response - blank_response
    all_blank = all_resp_mean - blank_response
    return peak_blank, all_blank


def do_sweep_mean(x):
    return len(x[x > 0.0])/2.0


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c
