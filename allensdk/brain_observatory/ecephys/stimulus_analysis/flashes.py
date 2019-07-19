import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from .stimulus_analysis import StimulusAnalysis, get_fr


class Flashes(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(Flashes, self).__init__(ecephys_session, **kwargs)

        self._colors = None
        self._col_color = 'Color'

        if self._params is not None:
            self._params = self._params['flashes']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'flash_250ms'


    @property
    def colors(self):
        if self._color is None:
            self._get_stim_table_stats()

        return self._colors
    

    @property
    def null_condition(self):
        return -1
    
    @property
    def METRICS_COLUMNS(self):
        return [('on_off_ratio_fl', np.float64), 
                ('sustained_idx_fl', np.float64),
                ('firing_rate_fl', np.float64), 
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

            metrics_df['on_off_ratio_fl'] = [self._get_on_off_ratio(unit) for unit in unit_ids]
            metrics_df['sustained_idx_fl'] = [self._get_sustained_index(unit) for unit in unit_ids]
            metrics_df['firing_rate_fl'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['reliability_fl'] = [self.get_reliability(unit, get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['fano_fl'] = [self.get_fano_factor(unit, get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_fl'] = [self.get_lifetime_sparesness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_fl', 'run_mod_fl']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        self._colors = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_color] != 'null'][self._col_color].unique())

