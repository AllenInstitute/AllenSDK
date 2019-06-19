from six import string_types
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi

from .ecephys_session import EcephysSession


class StimulusAnalysis(object):
    def __init__(self, ecephys_session, **kwargs):
        """
        :param ecephys_session: an EcephySession object or path to ece nwb file.
        """
        if isinstance(ecephys_session, EcephysSession):
            self._ecephys_session = ecephys_session
        elif isinstance(ecephys_session, string_types):
            nwb_version = kwargs.get('nwb_version', None)
            self._ecephys_session = EcephysSession.from_nwb_path(path=ecephys_session, nwb_version=nwb_version)

        self._cell_ids = None
        self._cells_filter = kwargs.get('filter', None)  # {'location': 'probeC', 'structure_acronym': 'VISp'}
        self._number_cells = None
        self._spikes = None
        self._stim_table = None
        self._stim_table_spontaneous = None
        self._stimulus_names = None
        self._running_speed = None
        self._sweep_events = None
        self._mean_sweep_events = None
        self._sweep_p_values = None
        self._peak = None

    @property
    def ecephys_session(self):
        return self._ecephys_session

    @property
    def cell_id(self):
        """Returns a list of unit-ids for which to apply the analysis"""
        # BOb analog
        if self._cell_ids is None:
            # Original analysis files was hardcoded that only cells from probeC/VISp, replaced with a filter dict.
            # TODO: Remove filter if it's unnessecary
            units_df = self.ecephys_session.units
            if self._cells_filter:
                mask = True
                for col, val in self._cells_filter.items():
                    mask &= units_df[col] == val
                units_df = units_df[mask]
            self._cell_ids = units_df.index.values

        return self._cell_ids

    @property
    def numbercells(self):
        """Get the number of units/cells."""
        # BOb analog
        if not self._number_cells:
            self._number_cells = len(self.cell_id)
        return self._number_cells

    @property
    def spikes(self):
        """Returns a diction unit_id -> spike-times."""
        if self._spikes:
            return self._spikes
        else:
            self._spikes = self.ecephys_session.spike_times
            if len(self._spikes) > self.numbercells:
                # if a filter has been applied such that not all the cells are being used in the analysis
                self._spikes = {k: v for k, v in self._spikes.items() if k in self.cell_id}

        return self._spikes

    @property
    def dxcm(self):
        """Returns an array of session running-speed velocities"""
        # BOb analog
        return self.ecephys_session.running_speed.values

    @property
    def dxtime(self):
        """Returns an array of session running speed timestamps"""
        # BOb analog
        return self._ecephys_session.running_speed.timestamps

    @property
    def stim_table(self):
        raise NotImplementedError()

    @property
    def stim_table_spontaneous(self):
        """Returns a stimulus table with only 'spontaneous' stimulus selected."""
        # BOb analog
        # Used by sweep_p_events for creating null dist.
        if self._stim_table_spontaneous is None:
            # TODO: The original version filtered out stims of len < 100, figure out why or if this value should
            #   be user-defined?
            stim_table = self.ecephys_session.get_presentations_for_stimulus(['spontaneous'])
            self._stim_table_spontaneous = stim_table[stim_table['duration'] > 100.0]

        return self._stim_table_spontaneous

    @property
    def sweep_events(self):
        """ Construct a dataframe describing events occuring within each stimulus presentation, by unit.

        Returns
        =======
        pd.DataFrame : 
            Each row is a stimulus presentation. Each column is a unit. Each cell contains a numpy array of 
            spike times occurring during that presentation (and emitted by that unit) relative to the onset of the 
            presentation (-1 second).

        """

        if self._sweep_events is None:
            start_times = self.stim_table['start_time'].values - 1.0  # TODO: 1 second pre?
            stop_times = self.stim_table['stop_time'].values
            sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=self.spikes.keys())

            for specimen_id, spikes in self.spikes.items(): # TODO: unit_id
                # In theory we should be able to use EcephysSession presentationwise_spike_times(). But ran into issues
                # with the "sides" certain boundary spikes will fall on, and will significantly affect the metrics
                # upstream.
                # TODO: what is going on here? instability? bug in ecephys_session? Kael remembers this occurring as a rate edge case
                start_indicies = np.searchsorted(spikes, start_times, side='left')
                stop_indicies = np.searchsorted(spikes, stop_times, side='right')

                sweep_events[specimen_id] = [spikes[start_indx:stop_indx] - start_times[indx] - 1.0 if stop_indx > start_indx else np.array([])
                                             for indx, (start_indx, stop_indx) in enumerate(zip(start_indicies, stop_indicies))]

            self._sweep_events = sweep_events

        return self._sweep_events

    @property
    def running_speed(self):
        if self._running_speed is None:
            stim_times = np.zeros(len(self.stim_table)*2, dtype=np.float64)
            stim_times[::2] = self.stim_table['start_time'].values
            stim_times[1::2] = self.stim_table['stop_time'].values
            sampled_indicies = np.where((self.dxtime >= stim_times[0])&(self.dxtime <= stim_times[-1]))[0]
            relevant_dxtimes = self.dxtime[sampled_indicies]
            relevant_dxcms = self.dxcm[sampled_indicies]

            indices = np.searchsorted(stim_times, relevant_dxtimes) - 1  # excludes dxtimes occuring at time_stop
            rs_tmp_df = pd.DataFrame({'running_speed': relevant_dxcms, 'stim_indicies': indices})

            # odd indicies have running speeds between start and stop times and should be removed
            rs_tmp_df = rs_tmp_df[rs_tmp_df['stim_indicies'].mod(2) == 0]

            # get averaged running speed for each stimulus
            rs_tmp_df = rs_tmp_df.groupby('stim_indicies').agg('mean')

            # some stimulus might not have an assoicated running_speed, set missing rows to NaN
            new_index = pd.Index(range(0, len(self.stim_table)*2, 2), name='stim_indicies')
            rs_tmp_df = rs_tmp_df.reindex(new_index)

            # reset index with presentation ids
            rs_tmp_df = rs_tmp_df.set_index(self.stim_table.index)
            self._running_speed = rs_tmp_df

        return self._running_speed

    @property
    def mean_sweep_events(self):
        """The mean values for sweep-events"""
        raise NotImplementedError()

    @property
    def sweep_p_values(self):
        """mean sweeps taken from randomized 'spontaneous' trial data."""
        if self._sweep_p_values is None:
            self._sweep_p_values = self.calc_sweep_p_values()

        return self._sweep_p_values

    @property
    def peak(self):
        """Returns a pandas DataFrame of the single-cell stimulus response metrics ranked by the cells unit-ids."""
        raise NotImplementedError()

    def calc_sweep_p_values(self, n_samples=10000, step_size=0.0001, offset=0.33):
        """ Calculates the probability, for each unit and stimulus presentation, that the number of spikes emitted by 
        that unit during that presentation could have been produced by that unit's spontaneous activity. This is 
        implemented as a permutation test using spontaneous activity (gray screen) periods as input data.

        Parameters
        ==========

        Returns
        =======
        sweep_p_values : pd.DataFrame
            Each row is a stimulus presentation. Each column is a unit. Cells contain the probability that the 
            unit's spontaneous activity could account for its observed spiking activity during that presentation
            (uncorrected for multiple comparisons).

        """

        # TODO: Code is currently a speed bottle-neck and could probably be improved.
        # Recreate the mean-sweep-table but using randomly selected 'spontaneuous' stimuli.
        shuffled_mean = np.empty((self.numbercells, n_samples))
        idx = np.random.choice(np.arange(self.stim_table_spontaneous['start_time'].iloc[0],
                                         self.stim_table_spontaneous['stop_time'].iloc[0],
                                         step_size), n_samples)  # TODO: what step size for np.arange?
        for shuf in range(n_samples):
            for i, v in enumerate(self.spikes.keys()):
                spikes = self.spikes[v]
                shuffled_mean[i, shuf] = len(spikes[(spikes > idx[shuf]) & (spikes < (idx[shuf] + offset))])

        sweep_p_values = pd.DataFrame(index=self.stim_table.index.values, columns=self.sweep_events.columns)
        for i, v in enumerate(self.spikes.keys()): # TODO: v -> unit_id
            subset = self.mean_sweep_events[v].values
            null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(subset), 1))
            actual_is_less = subset.reshape(len(subset), 1) <= null_dist_mat
            p_values = np.mean(actual_is_less, axis=1)
            sweep_p_values[v] = p_values

        return sweep_p_values

    def _get_reliability(self, specimen_id, st_mask):
        """computes trial-to-trial reliability of cell at its preferred condition

        :param specimen_id:
        :param st_mask:
        :return:
        """  # TODO: what are these?
        subset = self.sweep_events[st_mask][specimen_id].values
        subset += 1.0
        corr_matrix = np.empty((len(subset), len(subset)))
        for i in range(len(subset)):
            fri = get_fr(subset[i])
            for j in range(len(subset)):
                frj = get_fr(subset[j])
                # Warning: the pearson coefficient is likely to have a denominator of 0 for some cells/stimulus and give
                # a divide by 0 warning.
                r, p = st.pearsonr(fri[30:40], frj[30:40])
                corr_matrix[i, j] = r

        inds = np.triu_indices(len(subset), k=1)
        upper = corr_matrix[inds[0], inds[1]]
        return np.nanmean(upper)


def get_fr(spikes, num_timestep_second=30, filter_width=0.1):
    spikes = spikes.astype(float)
    spike_train = np.zeros((int(3.1*num_timestep_second)))  # TODO: hardcoded 3 second sweep length
    spike_train[(spikes*num_timestep_second).astype(int)] = 1
    filter_width = int(filter_width*num_timestep_second)
    fr = ndi.gaussian_filter(spike_train, filter_width)
    return fr