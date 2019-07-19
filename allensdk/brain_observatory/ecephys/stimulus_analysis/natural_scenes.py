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

        if self._params is not None:
            self._params = self._params['natural_scenes']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'Natural Images'

    @property
    def images(self):
        if self._images is None:
            self._get_stim_table_stats()

        return self._images

    @property
    def frames(self):
        # here to deal with naming difference between NWB 1 and 2
        return self.images

    @property
    def number_images(self):
        if self._images is None:
            self._get_stim_table_stats()

        return self._number_images

    @property
    def number_nonblank(self):
        # Some analysis function include -1 (119 values), others exlude it
        if self._number_nonblank is None:
            self._get_stim_table_stats()

        return self._number_nonblank

    @property
    def null_condition(self):
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

            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()

            metrics_df['pref_image_ns'] = [self.get_preferred_condition(unit) for unit in unit_ids]
            metrics_df['image_selectivity_ns'] = [self._get_image_selectivity(unit) for unit in unit_ids]
            metrics_df['firing_rate_ns'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['fano_ns'] = [self.get_fano_factor]
            metrics_df['time_to_peak_ns'] = [self.get_time_to_peak(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['reliability_ns'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_ns'] = [self.get_lifetime_sparesness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_dg', 'run_mod_dg']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        self._images = np.sort(self.stimulus_conditions[self._col_image].unique()).astype(np.int64)
        self._number_images = len(self._images)
        self._number_nonblank = len(self._images[self._images >= 0])


    def _get_image_selectivity(self, nc):
        """Calculates the image selectivity for cell

        # needs to be updated!

        :param nc:
        :return:
        """

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


    def _get_running_modulation(self, pref_image, v):
        """Computes running modulation of cell at its preferred condition provided there are at least 2 trials for both
        stationary and running conditions

        :param pref_image:
        :param v:
        :return: p_value of running modulation, running modulation metric, mean response to preferred condition when
        running, mean response to preferred condition when stationary
        """
        subset = self.mean_sweep_events[(self.stim_table['Image'] == pref_image)]
        speed_subset = self.running_speed[(self.stim_table['Image'] == pref_image)]

        subset_run = subset[speed_subset.running_speed >= 1]
        subset_stat = subset[speed_subset.running_speed < 1]
        if np.logical_and(len(subset_run) > 1, len(subset_stat) > 1):
            run = subset_run[v].mean()
            stat = subset_stat[v].mean()
            if run > stat:
                run_mod = (run - stat)/run
            elif stat > run:
                run_mod = -1 * (stat - run)/stat
            else:
                run_mod = 0
            (_, p) = st.ttest_ind(subset_run[v], subset_stat[v], equal_var=False)
            return p, run_mod, run, stat
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN


def do_sweep_mean_shifted(x):
    return len(x[(x > 0.066) & (x < 0.316)])/0.25
