import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from .stimulus_analysis import StimulusAnalysis, get_fr


class DotMotion(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(DotMotion, self).__init__(ecephys_session, **kwargs)

        self._dirvals = None
        self._col_dir = 'Ori'

        self._speeds = None
        self._col_speed = 'Speed'

        if self._params is not None:
            self._params = self._params['dot_motion']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'motion_stimulus'


    @property
    def directions(self):
        if self._dirvals is None:
            self._get_stim_table_stats()

        return self._dirvals
    
    @property
    def speeds(self):
        if self._speeds is None:
            self._get_stim_table_stats()

        return self._speeds

    @property
    def null_condition(self):
        return -1

    @property
    def METRICS_COLUMNS(self):
        return [('pref_speed_dm', np.float64), 
                ('pref_dir_dm', np.float64), 
                ('firing_rate_dm', np.float64), 
                ('fano_dm', np.float64),
                ('speed_tuning_idx_dm', np.float64), 
                ('time_to_peak_dm', np.float64), 
                ('reliability_dm', np.float64),
                ('lifetime_sparseness_dm', np.float64), 
                ('run_mod_dm', np.float64), 
                ('run_pval_dm', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:

            unit_ids = self.unit_ids
        
            metrics_df = self.empty_metrics_table()

            metrics_df['pref_speed_dm'] = [self._get_pref_speed(unit) for unit in unit_ids]
            metrics_df['pref_dir_dm'] = [self._get_pref_dir(unit) for unit in unit_ids]
            metrics_df['firing_rate_dm'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['fano_dm'] = [self.get_fano_factor(unit, get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['speed_tuning_idx_dm'] = [self._get_speed_tuning_index(unit) for unit in unit_ids]
            metrics_df['reliability_dm'] = [self.get_reliability(unit, get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_dm'] = [self.get_lifetime_sparesness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_dm', 'run_mod_dm']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        self._dirvals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_dir] != 'null'][self._col_dir].unique())
        self._speeds = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_speed] != 'null'][self._col_speed].unique())
