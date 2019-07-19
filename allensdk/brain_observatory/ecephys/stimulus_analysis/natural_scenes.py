import numpy as np
import pandas as pd
from six import string_types
import scipy.stats as st

from .stimulus_analysis import StimulusAnalysis


class NaturalScenes(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(NaturalScenes, self).__init__(ecephys_session, **kwargs)

        self._images = None
        self._number_images = None
        self._number_nonblank = None  # does not include Image number = -1.
        self._mean_sweep_events = None
        self._response_events = None
        self._response_trials = None
        self._metrics = None

        self._col_image = 'Image'

        self._trial_duration = 0.25

        if self._params is not None:
            self._params = self._params['natural_scenes']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'Natural Images'

        self._module_name = 'Natural Scenes'

    @property
    def images(self):
        """ Array of iamge labels """
        if self._images is None:
            self._get_stim_table_stats()

        return self._images

    @property
    def frames(self):
        # Required to deal with naming difference between NWB 1 and 2
        return self.images

    @property
    def number_images(self):
        """ Number of images shown """
        if self._images is None:
            self._get_stim_table_stats()

        return self._number_images

    @property
    def number_nonblank(self):
        """ Number of images shown (excluding blank condition) """
        if self._number_nonblank is None:
            self._get_stim_table_stats()

        return self._number_nonblank

    @property
    def null_condition(self):
        """ Stimulus condition ID for null (blank) stimulus """
        return self.stimulus_conditions[self.stimulus_conditions[self._col_image] == -1].index

    @property
    def METRICS_COLUMNS(self):
        return [('pref_image_ns', np.uint64), 
                ('image_selectivity_ns', np.float64), 
                ('firing_rate_ns', np.float64), 
                ('fano_ns', np.float64),
                ('time_to_peak_ns', np.float64), 
                ('reliability_ns', np.float64),
                ('lifetime_sparseness_ns', np.float64), 
                ('run_pval_ns', np.float64), 
                ('run_mod_ns', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:

            print('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            print(self.null_condition)

            metrics_df['pref_image_ns'] = [self.get_preferred_condition(unit) for unit in unit_ids]
            metrics_df['image_selectivity_ns'] = [self._get_image_selectivity(unit) for unit in unit_ids]
            metrics_df['firing_rate_ns'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['fano_ns'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['time_to_peak_ns'] = [self.get_time_to_peak(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['reliability_ns'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_ns'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_ns', 'run_mod_ns']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract image labels from the stimulus table """

        self._images = np.sort(self.stimulus_conditions[self._col_image].unique()).astype(np.int64)
        self._number_images = len(self._images)
        self._number_nonblank = len(self._images[self._images >= 0])


    def _get_image_selectivity(self, unit_id):
        """ Calculate the image selectivity for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        image_selectivity - metric

        """

        ### NEEDS TO BE UPDATED FOR NEW ADAPTER
        
        if False:

            fmin = self.response_events[1:, nc, 0].min()
            fmax = self.response_events[1:, nc, 0].max()
            rtj = np.empty((1000, 1))
            for j in range(1000):
                thresh = fmin + j*((fmax-fmin)/1000.)
                theta = np.empty((self.number_nonblank, 1))
                for im in range(self.number_nonblank):
                    if self.response_events[im+1, nc, 0] > thresh:  # im+1 to only look at images, not blanksweep
                        theta[im] = 1
                    else:
                        theta[im] = 0
                rtj[j] = theta.mean()
            biga = rtj.mean()
            return 1 - (2*biga)

        return np.nan
