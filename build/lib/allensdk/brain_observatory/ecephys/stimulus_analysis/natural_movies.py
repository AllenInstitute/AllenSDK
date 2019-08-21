import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit
import logging

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, get_fr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


class NaturalMovies(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the natural movies stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        nm_analysis = NaturalMovies(session)

    or, alternatively, pass in the file path::
        nm_analysis = Flashes('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        nm_analysis = NaturalMovies(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = nm_analysis.metrics()

    TODO: Need to find a default trial_duration otherwise class will fail
    """

    def __init__(self, ecephys_session, trial_duration=None, **kwargs):
        super(NaturalMovies, self).__init__(ecephys_session, trial_duration=trial_duration, **kwargs)

        self._metrics = None

        if self._params is not None:
            self._params = self._params['natural_movies']
            self._stimulus_key = self._params['stimulus_key']
        #else:
        #    self._stimulus_key = 'natural_movies'

        self._module_name = 'Natural Movies'


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
            logger.info('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            metrics_df['fano_nm'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['reliability_nm'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['firing_rate_nm'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['lifetime_sparseness_nm'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            run_vals = [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['run_pval_nm'] = [rv[0] for rv in run_vals]
            metrics_df['run_mod_nm'] = [rv[1] for rv in run_vals]
            #metrics_df.loc[:, ['run_pval_nm', 'run_mod_nm']] = \
            #        [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics

    @property
    def known_stimulus_keys(self):
        return ['natural_movies', 'natural_movie_1', 'natural_movie_3']


    def _get_stim_table_stats(self):

        pass


