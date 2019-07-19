import numpy as np
import pandas as pd
from six import string_types

from allensdk.brain_observatory.ecephys.stimulus_analysis import StimulusAnalysis
from allensdk.brain_observatory.ecephys.stimulus_analysis import get_reliability, get_lifetime_sparseness, \
    get_running_modulation


class NaturalScenes(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(NaturalScenes, self).__init__(ecephys_session, **kwargs)

        self._images = None
        self._number_images = None
        self._number_nonblank = None  # does not include Image number = -1.
        self._mean_sweep_events = None
        self._response_events = None
        self._response_trials = None
        self._peak = None

        self._col_image = 'Image'

        self._responsivness_threshold = kwargs.get('responsivness_threshold', 11)


    PEAK_COLS = [('cell_specimen_id', np.uint64), ('pref_image_ns', np.uint64), ('num_pref_trials_ns', np.uint64),
                 ('responsive_ns', bool), ('image_selectivity_ns', np.float64), ('reliability_ns', np.float64),
                 ('lifetime_sparseness_ns', np.float64), ('run_pval_ns', np.float64), ('run_mod_ns', np.float64),
                 ('run_resp_ns', np.float64), ('stat_resp_ns', np.float64)]

    @property
    def peak_columns(self):
        return [c[0] for c in self.PEAK_COLS]

    @property
    def peak_dtypes(self):
        return [c[1] for c in self.PEAK_COLS]

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
        # Some analysis function include -1 (119 values), others exclude it
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
    def peak(self):
        if self._peak is None:
            peak_df = pd.DataFrame(np.empty(self.numbercells, dtype=np.dtype(self.PEAK_COLS)),
                                   index=range(self.numbercells))

            peak_df['cell_specimen_id'] = list(self.spikes.keys())
            for nc, unit_id in enumerate(self.spikes.keys()):
                pref_image = np.where(self.response_events[1:, nc, 0] == self.response_events[1:, nc, 0].max())[0][0]
                peak_df.loc[nc, 'pref_image_ns'] = pref_image
                peak_df.loc[nc, 'num_pref_trials_ns'] = self.response_events[pref_image + 1, nc, 2]
                peak_df.loc[nc, 'responsive_ns'] = self.response_events[pref_image + 1, nc, 2] > self._responsivness_threshold

                peak_df.loc[nc, 'image_selectivity_ns'] = get_image_selectivity(self.response_events[:, nc, 0],
                                                                                self.number_nonblank)

                stim_table_mask = self.stim_table[self._col_image] == pref_image
                pref_sweeps = self.sweep_events[stim_table_mask][unit_id].values
                peak_df.loc[nc, 'reliability_ns'] = get_reliability(pref_sweeps, window_beg=30, window_end=40)

                subset = self.mean_sweep_events[stim_table_mask]
                speed_subset = self.running_speed[stim_table_mask]
                mse_subset_run = subset[speed_subset.running_speed >= 1][unit_id].values
                mse_subset_stat = subset[speed_subset.running_speed < 1][unit_id].values
                peak_df.loc[nc, ['run_pval_ns', 'run_mod_ns', 'run_resp_ns', 'stat_resp_ns']] = \
                    get_running_modulation(mse_subset_run, mse_subset_stat)

            peak_df['lifetime_sparseness_ns'] = get_lifetime_sparseness(self.response_events[1:, :, 0])

            self._peak = peak_df

        return self._peak

    def _get_stim_table_stats(self):
        stim_table = self.stim_table
        self._images = np.sort(stim_table[self._col_image].dropna().unique())
        # In NWB 2 the Image col is a float, but need them as ints for indexing
        self._images = self._images.astype(np.int64)
        self._number_images = len(self._images)
        self._number_nonblank = len(self._images[self._images >= 0])

    def _get_response_events(self):
        response_events = np.empty((self.number_images, self.numbercells, 3))

        # Ideally there should be the same # trials for each image. but just in case find the max
        max_n_images = int(self.stim_table[self._col_image].nunique())
        response_trials = np.empty((self.number_images, self.numbercells, max_n_images))
        response_trials[:] = np.nan

        for im in self.images:
            subset = self.mean_sweep_events[self.stim_table[self._col_image] == im]
            subset_p = self.sweep_p_values[self.stim_table[self._col_image] == im]
            response_events[im + 1, :, 0] = subset.mean(axis=0)
            response_events[im + 1, :, 1] = subset.std(axis=0) / np.sqrt(len(subset))
            response_events[im + 1, :, 2] = subset_p[subset_p < 0.05].count().values
            response_trials[im + 1, :, :subset.shape[0]] = subset.values.T

        self._response_trials = response_trials
        self._response_events = response_events


def get_image_selectivity(responses, number_nonblank):
    """Calculates the image selectivity for cell

    :param responses: An array of the mean cell responses across all images (including blank image).
    :param number_nonblank: Number of non-blank images
    :return:

    """
    fmin = responses[1:].min()
    fmax = responses[1:].max()
    normed_range = (fmax-fmin)/1000.0
    rtj = np.empty((1000, 1))
    for j in range(1000):
        thresh = fmin + j*normed_range  # ((fmax-fmin)/1000.)
        theta = np.empty((number_nonblank, 1))
        for im in range(number_nonblank):
            if responses[im+1] > thresh:  # im+1 to only look at images, not blanksweep
                theta[im] = 1
            else:
                theta[im] = 0
        rtj[j] = theta.mean()
    biga = rtj.mean()
    return 1 - (2*biga)


def do_sweep_mean_shifted(x, offset_lower=0.066, offset_upper=0.316):
    assert(offset_lower <= offset_upper)
    return len(x[(x > offset_lower) & (x < offset_upper)])/0.25
