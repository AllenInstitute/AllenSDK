import numpy as np
import pandas as pd
from six import string_types
import scipy.stats as st
import scipy.ndimage as ndi

from allensdk.brain_observatory.ecephys.stimulus_analysis import StimulusAnalysis


class NaturalScenes(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(NaturalScenes, self).__init__(ecephys_session, **kwargs)
        self._mean_sweep_events = None
        self._sweep_p_values = None
        self._response_events = None
        self._response_trials = None
        self._peak = None

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
    def sweep_p_values(self):
        if self._sweep_p_values is None:
            shuffled_mean = np.empty((self.numbercells, 10000))
            idx = np.random.choice(np.arange(self.stim_table_sp.start.iloc[0], self.stim_table_sp.end.iloc[0], 0.0001),
                                   10000)  # TODO: what step size for np.arange?
            # TODO: can we do this more efficiently
            for shuf in range(10000):
                for i, v in enumerate(self.spikes.keys()):
                    spikes = self.spikes[v]
                    shuffled_mean[i, shuf] = len(spikes[(spikes > idx[shuf]) & (spikes < (idx[shuf] + 0.33))])

            sweep_p_values = pd.DataFrame(index=self.stim_table.index.values, columns=self.sweep_events.columns)
            for i, v in enumerate(self.spikes.keys()):
                subset = self.mean_sweep_events[v].values
                null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(subset), 1))
                actual_is_less = subset.reshape(len(subset), 1) <= null_dist_mat
                p_values = np.mean(actual_is_less, axis=1)
                sweep_p_values[v] = p_values

        return self._sweep_p_values

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
            peak = pd.DataFrame(columns=('cell_specimen_id', 'pref_image_ns', 'num_pref_trials_ns', 'responsive_ns',
                                         'image_selectivity_ns', 'reliability_ns', 'lifetime_sparseness_ns',
                                         'run_pval_ns', 'run_mod_ns', 'run_resp_ns', 'stat_resp_ns'),
                                index=range(self.numbercells))
            peak['cell_specimen_id'] = self.spikes.keys()
            for nc, v in enumerate(self.spikes.keys()):
                pref_image = np.where(self.response_events[1:, nc, 0] == self.response_events[1:, nc, 0].max())[0][0]
                peak.pref_image_ns.iloc[nc] = pref_image
                peak.num_pref_trials_ns.iloc[nc] = self.response_events[pref_image + 1, nc, 2]
                if self.response_events[pref_image + 1, nc, 2] > 11:
                    peak.responsive_ns.iloc[nc] = True
                else:
                    peak.responsive_ns.iloc[nc] = False
                peak.image_selectivity_ns.iloc[nc] = self._get_image_selectivity(nc)
                peak.reliability_ns.iloc[nc] = self._get_reliability(pref_image, v)
                peak.run_pval_ns.iloc[nc], peak.run_mod_ns.iloc[nc], peak.run_resp_ns.iloc[nc], peak.stat_resp_ns.iloc[
                    nc] = self._get_running_modulation(pref_image, v)

            peak['lifetime_sparseness_ns'] = (
                        (1 - (1 / 118.) * ((np.power(self.response_events[:, :, 0].sum(axis=0), 2)) /
                                           (np.power(self.response_events[:, :, 0], 2).sum(axis=0)))) / (
                                    1 - (1 / 118.)))
            return peak

        return self._peak

    def _get_response_events(self):
        response_events = np.empty((119,self.numbercells,3))
        response_trials = np.empty((119,self.numbercells,50))
        response_trials[:] = np.NaN

        for im in range(-1,118):
            subset = self.mean_sweep_events[self.stim_table.frame==im]
            subset_p = self.sweep_p_values[self.stim_table.frame==im]
            response_events[im+1,:,0] = subset.mean(axis=0)
            response_events[im+1,:,1] = subset.std(axis=0)/np.sqrt(len(subset))
            response_events[im+1,:,2] = subset_p[subset_p<0.05].count().values
            response_trials[im+1,:,:subset.shape[0]] = subset.values.T

        self._response_trials = response_trials
        self._response_events = response_events

    def _get_image_selectivity(self, nc):
        """Calculates the image selectivity for cell

        :param nc:
        :return:
        """
        fmin = self.response_events[1:,nc,0].min()
        fmax = self.response_events[1:,nc,0].max()
        rtj = np.empty((1000,1))
        for j in range(1000):
            thresh = fmin + j*((fmax-fmin)/1000.)
            theta = np.empty((118,1))
            for im in range(118):
                if self.response_events[im+1,nc,0] > thresh:  #im+1 to only look at images, not blanksweep
                    theta[im] = 1
                else:
                    theta[im] = 0
            rtj[j] = theta.mean()
        biga = rtj.mean()
        return 1 - (2*biga)

    def _get_reliability(self, pref_image, v):
        """Computes trial-to-trial reliability of cell at its preferred condition

        :param pref_image:
        :param v:
        :return: reliability metric
        """
        subset = self.sweep_events[(self.stim_table.frame==pref_image)]
        subset += 1.
        corr_matrix = np.empty((len(subset),len(subset)))
        for i in range(len(subset)):
            fri = get_fr(subset[v].iloc[i])
            for j in range(len(subset)):
                frj = get_fr(subset[v].iloc[j])
                r,p = st.pearsonr(fri[30:40], frj[30:40])
                corr_matrix[i,j] = r

        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0],inds[1]]
        return np.nanmean(upper)

    def _get_running_modulation(self, pref_image, v):
        """Computes running modulation of cell at its preferred condition provided there are at least 2 trials for both
        stationary and running conditions

        :param pref_image:
        :param v:
        :return: p_value of running modulation, running modulation metric, mean response to preferred condition when
        running, mean response to preferred condition when stationary
        """
        subset = self.mean_sweep_events[(self.stim_table.frame==pref_image)]
        speed_subset = self.running_speed[(self.stim_table.frame==pref_image)]

        subset_run = subset[speed_subset.running_speed>=1]
        subset_stat = subset[speed_subset.running_speed<1]
        if np.logical_and(len(subset_run)>1, len(subset_stat)>1):
            run = subset_run[v].mean()
            stat = subset_stat[v].mean()
            if run > stat:
                run_mod = (run - stat)/run
            elif stat > run:
                run_mod = -1 * (stat - run)/stat
            else:
                run_mod = 0
            (_,p) = st.ttest_ind(subset_run[v], subset_stat[v], equal_var=False)
            return p, run_mod, run, stat
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN


def do_sweep_mean_shifted(x):
    return len(x[(x>0.066)&(x<0.316)])/0.25


def get_fr(spikes, num_timestep_second=30, filter_width=0.1):
    spikes = spikes.astype(float)
    spike_train = np.zeros((int(3.1*num_timestep_second))) #hardcoded 3 second sweep length
    spike_train[(spikes*num_timestep_second).astype(int)]=1
    filter_width = int(filter_width*num_timestep_second)
    fr = ndi.gaussian_filter(spike_train, filter_width)
    return fr
