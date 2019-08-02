import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit
import logging

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, osi, dsi, deg2rad
from ...circle_plots import FanPlotter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


class DriftingGratings(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the drifting gratings stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        dg_analysis = DriftingGratings(session)

    or, alternatively, pass in the file path::
        dg_analysis = DriftingGratings('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        dg_analysis = DriftingGratings(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = dg_analysis.metrics()

    """

    def __init__(self, ecephys_session, col_ori='Ori', col_tf='TF', trial_duration=2.0, **kwargs):
        super(DriftingGratings, self).__init__(ecephys_session, trial_duration=trial_duration, **kwargs)

        self._metrics = None

        self._orivals = None
        self._number_ori = None
        self._tfvals = None
        self._number_tf = None

        self._col_ori = col_ori
        self._col_tf = col_tf

        # self._trial_duration = trial_duration

        if self._params is not None:
            self._params = self._params['drifting_gratings']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'drifting_gratings'

        self._module_name = 'Drifting Gratings'

    @property
    def orivals(self):
        """ Array of grating orientation conditions """
        if self._orivals is None:
            self._get_stim_table_stats()

        return self._orivals

    @property
    def number_ori(self):
        """ Number of grating orientation conditions """
        if self._number_ori is None:
            self._get_stim_table_stats()

        return self._number_ori

    @property
    def tfvals(self):
        """ Array of grating temporal frequency conditions """
        if self._tfvals is None:
            self._get_stim_table_stats()

        return self._tfvals

    @property
    def number_tf(self):
        """ Number of grating temporal frequency conditions """
        if self._tfvals is None:
            self._get_stim_table_stats()

        return self._number_tf

    @property
    def null_condition(self):
        """ Stimulus condition ID for null (blank) stimulus """
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
            logger.info('Calculating metrics for ' + self.name)
        
            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            metrics_df['pref_ori_dg'] = [self._get_pref_ori(unit) for unit in unit_ids]
            metrics_df['pref_tf_dg'] = [self._get_pref_tf(unit) for unit in unit_ids]
            metrics_df['c50_dg'] = [self._get_c50(unit) for unit in unit_ids]
            metrics_df['f1_f0_dg'] = [self._get_f1_f0(unit) for unit in unit_ids]
            metrics_df['mod_idx_dg'] = [self._get_modulation_index(unit) for unit in unit_ids]
            metrics_df['g_osi_dg'] = [self._get_selectivity(unit, metrics_df.loc[unit]['pref_tf_dg'], 'osi') for unit in unit_ids]
            metrics_df['g_dsi_dg'] = [self._get_selectivity(unit, metrics_df.loc[unit]['pref_tf_dg'], 'dsi') for unit in unit_ids]
            metrics_df['firing_rate_dg'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['reliability_dg'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['fano_dg'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_dg'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_dg', 'run_mod_dg']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract orientations and temporal frequencies from the stimulus table """

        self._orivals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_ori] != 'null'][self._col_ori].unique())
        self._number_ori = len(self._orivals)

        self._tfvals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_tf] != 'null'][self._col_tf].unique())
        self._number_tf = len(self._tfvals)


    def _get_pref_ori(self, unit_id):

        """ Calculate the preferred orientation condition for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        pref_ori - stimulus orientation driving the maximal response

        """

        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[self._col_ori] == ori].tolist() for ori in self.orivals]
        df = pd.DataFrame(index=self.orivals,
                         data = {'spike_mean' : 
                                [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean() for condition_inds in similar_conditions]
                             }
                         ).rename_axis(self._col_ori)

        return df.idxmax().iloc[0]


    def _get_pref_tf(self, unit_id):

        """ Calculate the preferred temporal frequency condition for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        pref_tf - stimulus temporal frequency driving the maximal response

        """

        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[self._col_tf] == tf].tolist() for tf in self.tfvals]
        df = pd.DataFrame(index=self.tfvals,
                         data = {'spike_mean' : 
                                [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean() for condition_inds in similar_conditions]
                             }
                         ).rename_axis(self._col_tf)

        return df.idxmax().iloc[0]


    def _get_selectivity(self, unit_id, pref_tf, selectivity_type='osi'):

        """ Calculate the orientation or direction selectivity for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest
        pref_tf - preferred temporal frequency for this unit
        selectivity_type - 'osi' or 'dsi'

        Returns:
        -------
        selectivity - orientation or direction selectivity value

        """
        orivals_rad = deg2rad(self.orivals).astype('complex128')

        condition_inds = self.stimulus_conditions[self.stimulus_conditions[self._col_tf] == pref_tf].index.values
        df = self.conditionwise_statistics.loc[unit_id].loc[condition_inds]
        df = df.assign(Ori = self.stimulus_conditions.loc[df.index.values][self._col_ori])

        df = df.sort_values(by=['Ori'])

        tuning = np.array(df['spike_mean'].values)

        if selectivity_type == 'osi':
            return osi(orivals_rad, tuning)
        elif selectivity_type == 'dsi':
            return dsi(orivals_rad, tuning)



    def _get_f1_f0(self, unit_id):
        """ Calculate F1/F0 for a given unit

        A measure of how tightly locked a unit's firing rate is to the cycles of a drifting grating

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        f1_f0 - metric

        """

        return np.nan

    def _get_modulation_index(self, unit_id):
        """ Calculate modulation index for a given unit.

        Similar to F1/F0

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        modulation_index - metric

        """

        return np.nan

    def _get_c50(self, unit_id):
        """ Calculate C50 for a given unit.

        Only valid if the contrast tuning stimulus is present
        Otherwise, return NaN value

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        c50 - metric

        """

        return np.nan


    def _get_tfdi(self, unit_id, pref_ori):
        """ Calculate temporal frequency discrimination index for a given unit

        Only valid if the contrast tuning stimulus is present
        Otherwise, return NaN value

        Params:
        -------
        unit_id - unique ID for the unit of interest
        pref_ori - preferred orientation for that cell

        Returns:
        -------
        tfdi - metric

        """

        ### NEEDS TO BE UPDATED FOR NEW ADAPTER

        v = list(self.spikes.keys())[nc]
        tf_tuning = self.response_events[pref_ori, 1:, nc, 0]
        trials = self.mean_sweep_events[(self.stim_table['Ori'] == self.orivals[pref_ori])][v].values
        sse_part = np.sqrt(np.sum((trials-trials.mean())**2)/(len(trials)-5))
        return (np.ptp(tf_tuning))/(np.ptp(tf_tuning) + 2*sse_part)


    def _get_suppressed_contrast(self, unit_id, pref_ori, pref_tf):
        """ Calculate two metrics used to determine if a unit is suppressed by contrast

        Params:
        -------
        unit_id - unique ID for the unit of interest
        pref_ori - preferred orientation for that cell
        pref_tf - preferred temporal frequency for that cell

        Returns:
        -------
        peak_blank - metric
        all_blank - metric

        """

        ### NEEDS TO BE UPDATED FOR NEW ADAPTER

        blank = self.response_events[0, 0, nc, 0]
        peak = self.response_events[pref_ori, pref_tf+1, nc, 0]
        all_resp = self.response_events[:, 1:, nc, 0].mean()
        peak_blank = peak - blank
        all_blank = all_resp - blank
        
        return peak_blank, all_blank


    def _fit_tf_tuning(self, unit_id, pref_ori, pref_tf):

        """ Performs Gaussian or exponential fit on the temporal frequency tuning curve at the preferred orientation.

        Params:
        -------
        unit_id - unique ID for the unit of interest
        pref_ori - preferred orientation for that cell
        pref_tf - preferred temporal frequency for that cell

        Returns:
        -------
        fit_tf_ind - metric
        fit_tf - metric
        tf_low_cutoff - metric
        tf_high_cutoff - metric
        """

        ### NEEDS TO BE UPDATED FOR NEW ADAPTER

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

    ## VISUALIZATION ##

    def plot_raster(self, stimulus_condition_id, unit_id):
    
        """ Plot raster for one condition and one unit """

        idx_tf = np.where(self.tfvals == self.stimulus_conditions.loc[stimulus_condition_id][self._col_tf])[0]
        idx_ori = np.where(self.orivals == self.stimulus_conditions.loc[stimulus_condition_id][self._col_ori])[0]
        
        if len(idx_tf) == len(idx_ori) == 1:
     
            presentation_ids = \
                self.presentationwise_statistics.xs(unit_id, level=1)\
                [self.presentationwise_statistics.xs(unit_id, level=1)\
                ['stimulus_condition_id'] == stimulus_condition_id].index.values
            
            df = self.presentationwise_spike_times[ \
                (self.presentationwise_spike_times['stimulus_presentation_id'].isin(presentation_ids)) & \
                (self.presentationwise_spike_times['unit_id'] == unit_id) ]
                
            x = df.index.values - self.stim_table.loc[df.stimulus_presentation_id].start_time
            _, y = np.unique(df.stimulus_presentation_id, return_inverse=True) 
            
            plt.subplot(self.number_tf, self.number_ori, idx_tf*self.number_ori + idx_ori + 1)
            plt.scatter(x, y, c='k', s=1, alpha=0.25)
            plt.axis('off')


    def plot_response_summary(self, unit_id, bar_thickness=0.25):

        """ Plot the spike counts across conditions """
        df = self.stimulus_conditions.drop(index=self.null_condition)
    
        df['tf_index'] = np.searchsorted(self.tfvals, df[self._col_tf].values)
        df['ori_index'] = np.searchsorted(self.orivals, df[self._col_ori].values)
        
        cond_values = self.presentationwise_statistics.xs(unit_id, level=1)['stimulus_condition_id']
        
        x = df.loc[cond_values.values]['tf_index'] + np.random.rand(cond_values.size) * bar_thickness - bar_thickness/2
        y = self.presentationwise_statistics.xs(unit_id, level=1)['spike_counts']
        c = df.loc[cond_values.values]['tf_index']
        
        plt.subplot(2,1,1)
        plt.scatter(y,x,c=c,alpha=0.5,cmap='Purples',vmin=-5)
        locs, labels = plt.yticks(ticks=np.arange(self.number_tf), labels=self.tfvals)
        plt.ylabel('Temporal frequency')
        plt.xlabel('Spikes per trial')
        plt.ylim([self.number_tf,-1])

        x = df.loc[cond_values.values]['ori_index'] + np.random.rand(cond_values.size) * bar_thickness - bar_thickness/2
        y = self.presentationwise_statistics.xs(unit_id, level=1)['spike_counts']
        c = df.loc[cond_values.values]['ori_index']
        
        plt.subplot(2,1,2)
        plt.scatter(x,y,c=c,alpha=0.5,cmap='Spectral')
        locs, labels = plt.xticks(ticks=np.arange(self.number_ori), labels=self.orivals)
        plt.xlabel('Orientation')
        plt.ylabel('Spikes per trial')

    
    def make_fan_plot(self, unit_id):

        """ Make a 2P-style Fan Plot based on presentationwise spike counts"""

        angle_data = self.stimulus_conditions.loc[self.presentationwise_statistics.xs(unit_id, level=1)['stimulus_condition_id']][self._col_ori].values
        r_data = self.stimulus_conditions.loc[self.presentationwise_statistics.xs(unit_id, level=1)['stimulus_condition_id']][self._col_tf].values
        data = self.presentationwise_statistics.xs(unit_id, level=1)['spike_counts'].values
        
        null_trials = np.where(angle_data == 'null')[0]
        
        angle_data = np.delete(angle_data, null_trials)
        r_data = np.delete(r_data, null_trials)
        data = np.delete(data, null_trials)
        
        cmin = np.min(data)
        cmax = np.max(data)

        fp = FanPlotter.for_drifting_gratings()
        fp.plot(r_data = r_data, angle_data = angle_data, data =data, clim=[cmin, cmax])
        fp.show_axes(closed=False)
        plt.ylim([-5,5])
        plt.axis('equal')
        plt.axis('off')


### General functions ###

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c
