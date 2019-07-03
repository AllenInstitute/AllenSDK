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

        if self._params is not None:
            self.params = self._params['natural_scenes']

    METRICS_COLUMNS = [('unit_id', np.uint64), ('pref_image_ns', np.uint64), ('num_pref_trials_ns', np.uint64),
                 ('responsive_ns', bool), ('image_selectivity_ns', np.float64), ('reliability_ns', np.float64),
                 ('lifetime_sparseness_ns', np.float64), ('run_pval_ns', np.float64), ('run_mod_ns', np.float64),
                 ('run_resp_ns', np.float64), ('stat_resp_ns', np.float64)]

    @property
    def metrics_names(self):
        return [c[0] for c in self.METRICS_COLUMNS]

    @property
    def metrics_dtypes(self):
        return [c[1] for c in self.METRICS_COLUMNS]

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
        # Somee analysis function include -1 (119 values), others exlude it
        if self._number_nonblank is None:
            self._get_stim_table_stats()

        return self._number_nonblank

    @property
    def stim_table(self):
        # Stimulus table is already in EcephysSession object, just need to subselect 'static_gratings' presentations.
        if self._stim_table is None:
            # TODO: Give warning if no stimulus
            if self._stimulus_names is None:
                # Older versions of NWB files the stimulus name is in the form stimulus_gratings_N, so if
                # self._stimulus_names is not explicity specified try to figure out stimulus
                stims_table = self.ecephys_session.stimulus_presentations
                stim_names = [s for s in stims_table['stimulus_name'].unique()
                              if s.lower().startswith('natural_image') or s.lower().startswith('natural image')]

                self._stim_table = stims_table[stims_table['stimulus_name'].isin(stim_names)]

            else:
                self._stimulus_names = [self._stimulus_names] if isinstance(self._stimulus_names, string_types) \
                    else self._stimulus_names
                self._stim_table = self.ecephys_session.get_presentations_for_stimulus(self._stimulus_names)

        return self._stim_table

    @property
    def mean_sweep_events(self):
        if self._mean_sweep_events is None:
            self._mean_sweep_events = self.sweep_events.applymap(do_sweep_mean_shifted)

        return self._mean_sweep_events

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
    def metrics(self):
        if self._metrics is None:
            metrics_df = pd.DataFrame(np.empty(self.unit_count, dtype=np.dtype(self.METRICS_COLUMNS)),
                                   index=range(self.unit_count))

            metrics_df['cell_specimen_id'] = list(self.spikes.keys())
            for nc, unit_id in enumerate(self.spikes.keys()):
                pref_image = np.where(self.response_events[1:, nc, 0] == self.response_events[1:, nc, 0].max())[0][0]
                metrics_df.loc[nc, 'pref_image_ns'] = pref_image
                metrics_df.loc[nc, 'num_pref_trials_ns'] = self.response_events[pref_image + 1, nc, 2]
                metrics_df.loc[nc, 'responsive_ns'] = self.response_events[pref_image + 1, nc, 2] > 11
                metrics_df.loc[nc, 'image_selectivity_ns'] = self._get_image_selectivity(nc)

                stim_table_mask = self.stim_table['Image'] == pref_image
                metrics_df.loc[nc, 'reliability_ns'] = self._get_reliability(unit_id, stim_table_mask)
                metrics_df.loc[nc, ['run_pval_ns', 'run_mod_ns', 'run_resp_ns', 'stat_resp_ns']] = \
                    self._get_running_modulation(pref_image, unit_id)

            coeff_p = 1.0/float(self.number_nonblank)  # 1 - 1/18
            resp_means = self.response_events[:, :, 0]
            metrics_df['lifetime_sparseness_ns'] = (1 - coeff_p*((np.power(resp_means.sum(axis=0), 2)) /
                                                              (np.power(resp_means, 2).sum(axis=0)))) / (1.0 - coeff_p)

            self._metrics = metrics_df

        return self._metrics

    def _get_stim_table_stats(self):
        stim_table = self.stim_table
        self._images = np.sort(stim_table['Image'].dropna().unique())
        # In NWB 2 the Image col is a float, but need them as ints for indexing
        self._images = self._images.astype(np.int64)
        self._number_images = len(self._images)
        self._number_nonblank = len(self._images[self._images >= 0])

    def _get_response_events(self):
        response_events = np.empty((self.number_images, self.unit_count, 3))
        response_trials = np.empty((self.number_images, self.unit_count, 50))
        response_trials[:] = np.nan

        for im in self.images:
            subset = self.mean_sweep_events[self.stim_table['Image'] == im]
            subset_p = self.sweep_p_values[self.stim_table['Image'] == im]
            response_events[im + 1, :, 0] = subset.mean(axis=0)
            response_events[im + 1, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
            response_events[im + 1, :, 2] = subset_p[subset_p < 0.05].count().values
            response_trials[im + 1, :, :subset.shape[0]] = subset.values.T

        self._response_trials = response_trials
        self._response_events = response_events

    def _get_image_selectivity(self, nc):
        """Calculates the image selectivity for cell

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
