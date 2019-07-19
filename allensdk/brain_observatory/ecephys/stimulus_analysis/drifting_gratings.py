import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from .stimulus_analysis import StimulusAnalysis, get_fr


class DriftingGratings(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(DriftingGratings, self).__init__(ecephys_session, **kwargs)

        self._metrics = None

        self._orivals = None
        self._number_ori = None
        self._tfvals = None
        self._number_tf = None
        self._response_events = None
        self._response_trials = None

        self._col_ori = 'Ori'
        self._col_tf = 'TF'

        if self._params is not None:
            self._params = self._params['drifting_gratings']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'drifting_gratings'

    @property
    def metrics_names(self):
        return [c[0] for c in self.METRICS_COLUMNS]

    @property
    def metrics_dtypes(self):
        return [c[1] for c in self.METRICS_COLUMNS]

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
    def null_condition(self):
        return self.stimulus_conditions[self.stimulus_conditions[self._col_tf] == 'null'].index
    
    @property
    def METRICS_COLUMNS(self):
        return [('pref_ori_dg', np.float64), 
                ('pref_tf_dg', np.float64), 
                ('c50_dg', np.float64),
                ('f1_f0_dg', np.float64), 
                ('mod_idx_dg', np.float64),
                ('g_osi_dg', np.float64), 
                ('g_dsi_dg', np.float64), 
                ('firing_rate_dg', np.float64), 
                ('reliability_dg', np.float64),
                ('fano_dg', np.float64), 
                ('lifetime_sparseness_dg', np.float64), 
                ('run_pval_dg', np.float64),
                ('run_mod_dg', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:
        
            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            metrics_df['pref_ori_dg'] = [self._get_pref_ori(unit) for unit in unit_ids]
            metrics_df['pref_tf_dg'] = [self._get_pref_tf(unit) for unit in unit_ids]
            metrics_df['c50_dg'] = np.nan
            metrics_df['f1_f0_dg'] = [self._get_f1_f0(unit) for unit in unit_ids]
            metrics_df['mod_idx_dg'] = [self._get_modulation_index(unit) for unit in unit_ids]
            metrics_df['g_osi_dg'] = [self._get_selectivity(unit, metrics_df.loc[unit]['pref_tf_dg'], 'osi') for unit in unit_ids]
            metrics_df['g_dsi_dg'] = [self._get_selectivity(unit, metrics_df.loc[unit]['pref_tf_dg'], 'dsi') for unit in unit_ids]
            metrics_df['firing_rate_dg'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['reliability_dg'] = [self._get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['fano_dg'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_dg'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_dg', 'run_mod_dg']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        self._orivals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_ori] != 'null'][self._col_ori].unique())
        self._number_ori = len(self._orivals)

        self._tfvals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_tf] != 'null'][self._col_tf].unique())
        self._number_tf = len(self._tfvals)


    def _get_pref_ori(self, unit_id):

        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[self._col_tf] == tf].tolist() for tf in self.tfvals]
        df = pd.DataFrame(index=self.tfvals,
                         data = {'spike_mean' : 
                                [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean() for condition_inds in similar_conditions]
                             }
                         ).rename_axis(self._col_tf)

        return df.idxmax()

    def _get_pref_tf(self, unit_id):

        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[self._col_ori] == ori].tolist() for ori in self.orivals]
        df = pd.DataFrame(index=self.orivals,
                         data = {'spike_mean' : 
                                [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean() for condition_inds in similar_conditions]
                             }
                         ).rename_axis(self._col_ori)

        return df.idxmax()




    def _get_selectivity(self, unit_id, pref_tf, selectivity_type='osi'):
        """computes orientation and direction selectivity (cv) for a particular unit

        :param unit_id: ID for the unit of interest
        :param pref_tf: preferred temporal frequency for this unit
        :return:
        """
        orivals_rad = np.deg2rad(self.orivals)
        
        condition_inds = self.stimulus_conditions[self.stimulus_conditions[self._col_tf] == pref_tf].index.values
        df = self.conditionwise_statistics.loc[unit_id].loc[condition_inds]
        df = df.assign(Ori = self.stimulus_conditions.loc[df.index.values][self._col_ori])
        df = df.sort_values(by=[self._col_ori])

        tuning = df['spike_mean'].values

        if selectivity_type == 'osi':
            return _osi(orivals_rad, tuning)
        elif selectivity_type == 'dsi':
            return _dsi(orivals_rad, tuning)


    def _osi(self, orivals, tuning):
        """Computes orientation selectivity for a tuning curve 

        """

        cv_top = tuning * np.exp(1j * 2 * orivals)
        return np.abs(cv_top.sum()) / tuning.sum()


    def _dsi(self, orivals, tuning):
        """Computes direction selectivity for a tuning curve 

        """

        cv_top = tuning * np.exp(1j * orivals)
        return np.abs(cv_top.sum()) / tuning.sum()


    def _get_reliability(self, unit_id, preferred_condition):
        """Computes trial-to-trial reliability of units at their preferred condition

        :param pref_ori:
        :param pref_tf:
        :param v:
        :return:
        """

        subset = self.presentationwise_statistics[
                    self.presentationwise_statistics['stimulus_condition_id'] == preferred_condition
                    ].xs(unit_id, level=1)['spike_counts'].values
        subset += 1

        corr_matrix = np.empty((len(subset), len(subset)))
        for i in range(len(subset)):
            fri = get_fr(subset[i])
            for j in range(len(subset)):
                frj = get_fr(subset[j])
                # TODO: Is there a reason this method get fr[30:] and the another stim analysis classes gets fr[30:40]?
                #    We could consolidate this method across all the classes.
                r, p = st.pearsonr(fri[30:], frj[30:])
                corr_matrix[i, j] = r

        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0], inds[1]]
        return np.nanmean(upper)


    def _get_tfdi(self, pref_ori, nc):
        """Computes temporal frequency discrimination index for cell

        :param pref_ori:
        :param nc:
        :return: tf discrimination index
        """
        v = list(self.spikes.keys())[nc]
        tf_tuning = self.response_events[pref_ori, 1:, nc, 0]
        trials = self.mean_sweep_events[(self.stim_table['Ori'] == self.orivals[pref_ori])][v].values
        sse_part = np.sqrt(np.sum((trials-trials.mean())**2)/(len(trials)-5))
        return (np.ptp(tf_tuning))/(np.ptp(tf_tuning) + 2*sse_part)

    def _get_suppressed_contrast(self, pref_ori, pref_tf, nc):
        """Computes two metrics to be used to identify cells that are suppressed by contrast

        :param pref_ori:
        :param pref_tf:
        :param nc:
        :return:
        """
        blank = self.response_events[0, 0, nc, 0]
        peak = self.response_events[pref_ori, pref_tf+1, nc, 0]
        all_resp = self.response_events[:, 1:, nc, 0].mean()
        peak_blank = peak - blank
        all_blank = all_resp - blank
        return peak_blank, all_blank

    def _fit_tf_tuning(self, pref_ori, pref_tf, nc):
        """Performs gaussian or exponential fit on the temporal frequency tuning curve at preferred orientation.

        :param pref_ori:
        :param pref_tf:
        :param nc:
        :return: index for the preferred tf from the curve fit, prefered tf from the curve fit, low cutoff tf from the
        curve fit, high cutoff tf from the curve fit
        """
        tf_tuning = self.response_events[pref_ori, 1:, nc, 0]
        fit_tf_ind = np.NaN
        fit_tf = np.NaN
        tf_low_cutoff = np.NaN
        tf_high_cutoff = np.NaN
        if pref_tf in range(1, 4):
            try:
                popt, pcov = curve_fit(gauss_function, range(5), tf_tuning, p0=[np.amax(tf_tuning), pref_tf, 1.],
                                       maxfev=2000)
                tf_prediction = gauss_function(np.arange(0., 4.1, 0.1), *popt)
                fit_tf_ind = popt[1]
                fit_tf = np.power(2, popt[1])
                low_cut_ind = np.abs(tf_prediction - (tf_prediction.max() / 2.))[:tf_prediction.argmax()].argmin()
                high_cut_ind = np.abs(tf_prediction - (tf_prediction.max() / 2.))[
                               tf_prediction.argmax():].argmin() + tf_prediction.argmax()
                if low_cut_ind > 0:
                    low_cutoff = np.arange(0, 4.1, 0.1)[low_cut_ind]
                    tf_low_cutoff = np.power(2, low_cutoff)
                elif high_cut_ind < 49:
                    high_cutoff = np.arange(0, 4.1, 0.1)[high_cut_ind]
                    tf_high_cutoff = np.power(2, high_cutoff)
            except Exception:
                pass
        else:
            fit_tf_ind = pref_tf
            fit_tf = self.tfvals[pref_tf]
            try:
                popt, pcov = curve_fit(exp_function, range(5), tf_tuning,
                                       p0=[np.amax(tf_tuning), 2., np.amin(tf_tuning)], maxfev=2000)
                tf_prediction = exp_function(np.arange(0., 4.1, 0.1), *popt)
                if pref_tf == 0:
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


def do_sweep_mean(x):
    return len(x[x > 0.])/2.0


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c
