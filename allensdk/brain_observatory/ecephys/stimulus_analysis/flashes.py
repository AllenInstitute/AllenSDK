import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, get_fr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Flashes(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the full-field flash stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        fl_analysis = Flashes(session)

    or, alternatively, pass in the file path::
        fl_analysis = Flashes('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        fl_analysis = Flashes(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = fl_analysis.metrics()

    """

    def __init__(self, ecephys_session, **kwargs):
        super(Flashes, self).__init__(ecephys_session, **kwargs)

        self._metrics = None

        self._colors = None
        self._col_color = 'color'

        if self._params is not None:
            self._params = self._params['flashes']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'flashes'

        self._trial_duration = 0.25

        self._module_name = 'Flashes'


    @property
    def colors(self):
        """ Array of 'color' conditions (black vs. white flash) """
        if self._colors is None:
            self._get_stim_table_stats()

        return self._colors

    @property
    def number_colors(self):
        """ Number of 'color' conditions (black vs. white flash) """
        if self._colors is None:
            self._get_stim_table_stats()

        return len(self._colors)
    

    @property
    def null_condition(self):
        """ Stimulus condition ID for null stimulus (not used, so set to -1) """
        return -1
    
    @property
    def METRICS_COLUMNS(self):
        return [('on_off_ratio_fl', np.float64), 
                ('sustained_idx_fl', np.float64),
                ('firing_rate_fl', np.float64), 
                ('time_to_peak_fl', np.float64), 
                ('reliability_fl', np.float64),
                ('fano_fl', np.float64), 
                ('lifetime_sparseness_fl', np.float64), 
                ('run_pval_fl', np.float64),
                ('run_mod_fl', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:

            unit_ids = self.unit_ids
        
            metrics_df = self.empty_metrics_table()

            if len(self. stim_table) > 0:

                print('Calculating metrics for ' + self.name)

                metrics_df['on_off_ratio_fl'] = [self._get_on_off_ratio(unit) for unit in unit_ids]
                metrics_df['sustained_idx_fl'] = [self._get_sustained_index(unit) for unit in unit_ids]
                metrics_df['firing_rate_fl'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
                metrics_df['reliability_fl'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
                metrics_df['time_to_peak_fl'] = [self.get_time_to_peak(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
                metrics_df['fano_fl'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
                metrics_df['lifetime_sparseness_fl'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
                metrics_df.loc[:, ['run_pval_fl', 'run_mod_fl']] = \
                        [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract colors from the stimulus table """

        self._colors = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_color] != 'null'][self._col_color].unique())


    def _get_sustained_index(self, unit_id):

        """ Calculate the sustained index for a given unit, a measure of the transience of
        the flash response.

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        sustained_index - metric

        """

        return np.nan

    def _get_on_off_ratio(self, unit_id):

        """ Calculate the ratio of the on response vs. off response for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        on_off_ratio - metric

        """

        on_condition_id = self.stimulus_conditions[self.stimulus_conditions[self._col_color] == '1.0'].index.values
        off_condition_id = self.stimulus_conditions[self.stimulus_conditions[self._col_color] == '-1.0'].index.values

        on_mean_spikes = self.conditionwise_statistics.loc[unit_id].loc[on_condition_id]['spike_mean'].values
        off_mean_spikes = self.conditionwise_statistics.loc[unit_id].loc[off_condition_id]['spike_mean'].values

        if off_mean_spikes[0] > 0:
            return on_mean_spikes[0] / off_mean_spikes[0]
        else:
            return np.nan


    ## VISUALIZATION ##

    def plot_raster(self, stimulus_condition_id, unit_id):
    
        """ Plot raster for one condition and one unit """

        idx_color = np.where(self.colors == self.stimulus_conditions.loc[stimulus_condition_id][self._col_color])[0]

        if len(idx_color) == 1:
     
            presentation_ids = \
                self.presentationwise_statistics.xs(unit_id, level=1)\
                [self.presentationwise_statistics.xs(unit_id, level=1)\
                ['stimulus_condition_id'] == stimulus_condition_id].index.values
            
            df = self.presentationwise_spike_times[ \
                (self.presentationwise_spike_times['stimulus_presentation_id'].isin(presentation_ids)) & \
                (self.presentationwise_spike_times['unit_id'] == unit_id) ]
                
            x = df.index.values - self.stim_table.loc[df.stimulus_presentation_id].start_time
            _, y = np.unique(df.stimulus_presentation_id, return_inverse=True) 
            
            plt.subplot(self.number_colors, 1, idx_color + 1)
            plt.scatter(x, y, c='k', s=1, alpha=0.25)
            plt.axis('off')


    def plot_response(self, unit_id):

        """ Plot a histogram for the two conditions """

        plot_colors = ('darkslateblue', 'grey')

        for idx, color in enumerate(self.colors):

            condition_id = self.stimulus_conditions[self.stimulus_conditions['color'] == color].index.values[0]
            
            psth = self.conditionwise_psth.sel(unit_id=unit_id, stimulus_condition_id =condition_id).values
            
            plt.bar(np.arange(len(psth))-0.5, psth, color=plot_colors[idx], alpha=0.5, width=1.0)
            plt.step(np.arange(len(psth)), psth, color=plot_colors[idx])
            plt.axis('off')
    