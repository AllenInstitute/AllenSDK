import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from .stimulus_analysis import StimulusAnalysis, get_fr


class NaturalMovies(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(NaturalMovies, self).__init__(ecephys_session, **kwargs)

        if self._params is not None:
            self._params = self._params['natural_movies']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'natural_movies'


    @property
    def null_condition(self):
        return -1
    

    @property
    def METRICS_COLUMNS(self):
        return [('fano_nm', np.uint64), 
                ('reliability_nm', np.float64),
                ('firing_rate_nm', np.float64),
                ('lifetime_sparseness_nm', np.float64), 
                ('run_pval_ns', np.float64), 
                ('run_mod_ns', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:

            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            metrics_df['fano_nm'] = [self.get_fano_factor(unit) for unit in unit_ids]
            metrics_df['reliability_nm'] = [self.get_reliability(unit) for unit in unit_ids]
            metrics_df['firing_rate_ns'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['lifetime_sparseness_ns'] = [self.get_lifetime_sparesness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_dg', 'run_mod_dg']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        pass


