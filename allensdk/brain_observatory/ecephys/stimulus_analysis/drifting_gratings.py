import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.fftpack import fft

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, osi, dsi, deg2rad
from ...circle_plots import FanPlotter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

    def __init__(self, ecephys_session, **kwargs):
        super(DriftingGratings, self).__init__(ecephys_session, **kwargs)

        self._metrics = None

        self._orivals = None
        self._number_ori = None
        self._tfvals = None
        self._number_tf = None

        self._col_ori = 'ori'
        self._col_tf = 'TF'
        self._col_contrast = 'contrast'

        self._trial_duration = 2.0

        if self._params is not None:
            self._params = self._params['drifting_gratings']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'drifting_gratings'

        self._module_name = 'Drifting Gratings'

        stim_table = self.stim_table

        self._stim_table_contrast = stim_table[stim_table['stimulus_name'] == 'drifting_gratings_contrast']
        self._stim_table = stim_table[stim_table['stimulus_name'] != 'drifting_gratings_contrast']

        self._conditionwise_statistics_contrast = None
        self._stimulus_conditions_contrast = None

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
    def contrastvals(self):
        """ Array of grating temporal frequency conditions """
        if self._contrast_vals is None:
            self._get_stim_table_stats()

        return self._contrast_vals

    @property
    def number_contrast(self):
        """ Number of grating temporal frequency conditions """
        if self._number_contrast is None:
            self._get_stim_table_stats()

        return self._number_contrast

    @property
    def null_condition(self):
        """ Stimulus condition ID for null (blank) stimulus """
        return self.stimulus_conditions[self.stimulus_conditions[self._col_tf] == 'null'].index

    @property
    def stimulus_conditions_contrast(self):
        """ Stimulus conditions for contrast stimulus """
        if self._stimulus_conditions_contrast is None:
            contrast_condition_list = self._stim_table_contrast.stimulus_condition_id.unique()

            self._stimulus_conditions_contrast = \
                    self.ecephys_session.stimulus_conditions[
                        self.ecephys_session.stimulus_conditions.index.isin(contrast_condition_list)
                    ]

        return self._stimulus_conditions_contrast

    @property
    def conditionwise_statistics_contrast(self):
        """ Conditionwise statistics for contrast stimulus """
        if self._conditionwise_statistics_contrast is None:

            self._conditionwise_statistics_contrast = \
                    self.ecephys_session.conditionwise_spike_statistics(self._stim_table_contrast.index.values,
                        self.unit_ids)

        return self._conditionwise_statistics_contrast

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

            print('Calculating metrics for ' + self.name)
        
            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            if len(self.stim_table) > 0:

                metrics_df['pref_ori_dg'] = [self._get_pref_ori(unit) for unit in unit_ids]
                metrics_df['pref_tf_dg'] = [self._get_pref_tf(unit) for unit in unit_ids]
                metrics_df['f1_f0_dg'] = [self._get_f1_f0(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
                metrics_df['mod_idx_dg'] = [self._get_modulation_index(unit) for unit in unit_ids]
                metrics_df['g_osi_dg'] = [self._get_selectivity(unit, metrics_df.loc[unit]['pref_tf_dg'], 'osi') for unit in unit_ids]
                metrics_df['g_dsi_dg'] = [self._get_selectivity(unit, metrics_df.loc[unit]['pref_tf_dg'], 'dsi') for unit in unit_ids]
                metrics_df['firing_rate_dg'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
                metrics_df['reliability_dg'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
                metrics_df['fano_dg'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
                metrics_df['lifetime_sparseness_dg'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
                metrics_df.loc[:, ['run_pval_dg', 'run_mod_dg']] = \
                        [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            if len(self._stim_table_contrast) > 0:
                metrics_df['c50_dg'] = [self._get_c50(unit) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract orientations and temporal frequencies from the stimulus table """

        self._orivals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_ori] != 'null'][self._col_ori].unique())
        self._number_ori = len(self._orivals)

        self._tfvals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_tf] != 'null'][self._col_tf].unique())
        self._number_tf = len(self._tfvals)

        self._contrastvals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_contrast] != 'null'][self._col_contrast].unique())
        self._number_contrast = len(self._contrastvals)


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
        if not 'drifting_gratings' in self.stim_table.stimulus_name.unique():
            return np.nan

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
        if not 'drifting_gratings' in self.stim_table.stimulus_name.unique():
            return np.nan

        orivals_rad = deg2rad(self.orivals).astype('complex128')

        condition_inds = self.stimulus_conditions[self.stimulus_conditions[self._col_tf] == pref_tf].index.values
        df = self.conditionwise_statistics.loc[unit_id].loc[condition_inds]
        df = df.assign(ori = self.stimulus_conditions.loc[df.index.values][self._col_ori])
        df = df.sort_values(by=[self._col_ori])

        tuning = np.array(df['spike_mean'].values)

        if selectivity_type == 'osi':
            return osi(orivals_rad, tuning)
        elif selectivity_type == 'dsi':
            return dsi(orivals_rad, tuning)



    def _get_f1_f0(self, unit_id, condition_id):
        """ Calculate F1/F0 for a given unit

        A measure of how tightly locked a unit's firing rate is to the cycles of a drifting grating

        Params:
        -------
        unit_id - unique ID for the unit of interest
        condition_id - ID for the condition of interest (usually the preferred condition)

        Returns:
        -------
        f1_f0 - metric

        """

        presentation_ids = self.stim_table[self.stim_table['stimulus_condition_id'] == 
                                              condition_id].index.values
                                              
        tf = self.stim_table.loc[presentation_ids[0]][self._col_tf]

        dataset = self.ecephys_session.presentationwise_spike_counts(bin_edges = np.arange(0, self._trial_duration, 0.001),
                                                                  stimulus_presentation_ids = presentation_ids,
                                                                  unit_ids = unit_id
                                                                  ).drop('unit_id')
        arr = np.squeeze(dataset['spike_counts'].values)

        return f1_f0(arr, tf)


    def _get_modulation_index(self, unit_id, condition_id):
        """ Calculate modulation index for a given unit.

        Params:
        -------
        unit_id - unique ID for the unit of interest
        condition_id - ID for the condition of interest (usually the preferred condition)

        Returns:
        -------
        modulation_index - metric

        """

        tf = self.stimulus_conditions.loc[condition_id][self._col_tf]

        data = self.conditionwise_psth.sel(unit_id = unit_id, stimulus_condition_id=condition_id).data 
        sample_rate = 1 / np.mean(np.diff(self.conditionwise_psth.time_relative_to_stimulus_onset))

        return modulation_index(data, tf, sample_rate)


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

        contrast_conditions = self._stim_table_contrast[(self._stim_table_contrast[self._col_ori] == self._get_pref_ori(unit_id))]['stimulus_condition_id'].unique()

        contrasts = self.stimulus_conditions_contrast.loc[contrast_conditions][self._col_contrast].values.astype('float')
        mean_responses = self.conditionwise_statistics_contrast.loc[unit_id].loc[contrast_conditions]['spike_mean'].values.astype('float')

        return c50(contrasts, mean_responses)



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

    
    def make_star_plot(self, unit_id):

        """ Make a 2P-style Star Plot based on presentationwise spike counts"""

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
    """
    fit gaussian function at log scale
    good for fitting band pass, not good at low pass or high pass
    """

    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def exp_function(x, a, b, c):
    return a*np.exp(-b*x)+c

def contrast_curve(x,b,c,d,e):
    """
     fit sigmoid function at log scale
     not good for fitting band pass
     - b: hill slope
     - c: min response
     - d: max response
     - e: EC50
    """
    return (c+(d-c)/(1+np.exp(b*(np.log(x)-np.log(e)))))


def c50(x,y):

    """
    Computes C50 of a contrast response function

    Parameters:
    -----------
    x : array of contrast values
    y : array of responses (spike rates)

    Returns:
    --------
    c50 : metric

    """

    try:
        fitCoefs, covMatrix = curve_fit(contrast_curve, x, y, maxfev = 100000)
    except RuntimeError:
        return np.nan
    
    resids = y-contrast_curve(x.astype('float'),*fitCoefs)
    
    X = np.linspace(min(x)*0.9,max(x)*1.1,256)
    y_fit = contrast_curve(X,*fitCoefs)
    
    y_middle = (max(y_fit) - min(y_fit)) / 2 + min(y_fit)
    
    try:
        c50 = X[np.searchsorted(y_fit, y_middle)]
    except IndexError:
        return np.nan
        
    return c50


def f1_f0(arr, tf):

    """
    Computes F1/F0 of a drifting grating response

    Parameters:
    -----------
    arr : DataArray with trials x times
    tf : temporal frequency of the stimulus

    Returns:
    --------
    f1_f0 : metric

    """

    num_trials = dataset.stimulus_presentation_id.size
    num_bins = dataset.time_relative_to_stimulus_onset.size
    trial_duration = dataset.time_relative_to_stimulus_onset.max()

    cycles_per_trial = int(tf * trial_duration)

    bins_per_cycle = int(num_bins / cycles_per_trial)

    arr = arr[:, :cycles_per_trial*bins_per_cycle].reshape((num_trials, cycles_per_trial, bins_per_cycle))

    avg_rate = np.mean(arr,1)

    AMP = 2*np.abs(fft(avg_rate, bins_per_cycle)) / bins_per_cycle

    f0 = 0.5*AMP[:,0]
    f1 = AMP[:,1]
    selection = f0 > 0.0

    return np.nanmean(f1[selection]/f0[selection])


def modulation_index(response, tf, sample_rate):

    """  Depth of modulation by each cycle of a drifting grating; similar to F1/F0

    ref: Matteucci et al. (2019) Nonlinear processing of shape information 
         in rat lateral extrastriate cortex. J Neurosci 39: 1649-1670

    """

    f, psd = signal.welch(response, fs=sample_rate, nperseg=1024)

    tf_index = np.searchsorted(f, tf)

    MI = abs((psd[tf_index] - np.mean(psd))/np.sqrt(np.mean(psd**2)-np.mean(psd)**2))

    return MI
