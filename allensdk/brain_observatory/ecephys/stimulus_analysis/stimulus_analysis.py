from six import string_types
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi

from ..ecephys_session import EcephysSession


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

        self._unit_ids = None
        self._unit_filter = kwargs.get('filter', None)
        self._params = kwargs.get('params', None)
        self._unit_count = None
        self._stim_table = None
        self._conditionwise_statistics = None

        self._spikes = None
        self._stim_table_spontaneous = None
        self._stimulus_names = None
        self._running_speed = None
        self._sweep_events = None
        self._mean_sweep_events = None
        self._sweep_p_values = None
        self._metrics = None

        self._trial_duration = None



        # An padding of time to look back when gathering events belong to a given stimulus, used by sweep_events and
        # get_reliability
        self._sweep_pre_time = kwargs.get('sweep_pre_time', 1.0)

    @property
    def ecephys_session(self):
        return self._ecephys_session

    @property
    def unit_ids(self):
        """Returns a list of unit IDs for which to apply the analysis"""
        if self._unit_ids is None:
            units_df = self.ecephys_session.units
            if self._unit_filter:
                mask = True
                for col, val in self._unit_filter.items():
                    mask &= units_df[col] == val
                units_df = units_df[mask]
            self._unit_ids = units_df.index.values

        return self._unit_ids

    @property
    def unit_count(self):
        """Get the number of units."""
        if not self._unit_count:
            self._unit_count = len(self.unit_ids)
        return self._unit_count

    @property
    def spikes(self):
        """Returns a dictionary of unit_id -> spike-times."""
        if self._spikes:
            return self._spikes
        else:
            self._spikes = self.ecephys_session.spike_times
            if len(self._spikes) > self.unit_count:
                # if a filter has been applied such that not all the cells are being used in the analysis
                self._spikes = {k: v for k, v in self._spikes.items() if k in self.unit_ids}

        return self._spikes

    @property
    def dxcm(self):
        """Returns an array of session running-speed velocities"""
        return self.ecephys_session.running_speed.values

    @property
    def dxtime(self):
        """Returns an array of session running speed timestamps"""
        return self._ecephys_session.running_speed.timestamps

    @property
    def stim_table(self):
        raise NotImplementedError()

    @property
    def stim_table_spontaneous(self):
        """Returns a stimulus table with only 'spontaneous' stimulus selected."""
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
            start_times = self.stim_table['start_time'].values - self._sweep_pre_time
            stop_times = self.stim_table['stop_time'].values
            sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=self.spikes.keys())

            for unit_id, spikes in self.spikes.items():
                # In theory we should be able to use EcephysSession presentationwise_spike_times(). But ran into issues
                # with the "sides" certain boundary spikes will fall on, and will significantly affect the metrics
                # upstream.
                start_indicies = np.searchsorted(spikes, start_times, side='left')
                stop_indicies = np.searchsorted(spikes, stop_times, side='right')

                sweep_events[unit_id] = [spikes[start_indx:stop_indx] - start_times[indx] - self._sweep_pre_time if stop_indx > start_indx else np.array([])
                                             for indx, (start_indx, stop_indx) in enumerate(zip(start_indicies, stop_indicies))]

            self._sweep_events = sweep_events

        return self._sweep_events

    @property
    def conditionwise_statistics(self):
        """ Construct a dataframe with the statistics for each stimulus condition, by unit

        Returns
        =======
        pd.DataFrame :
            MultiIndex : unit_id, stimulus_condition_id
            Columns : spike_count, spike_mean, spike_sem, spike_std, stimulus_presentation_count

        """

        if self._conditionwise_statistics is None:

            self._conditionwise_statistics = \
                    self.ecephys_session.conditionwise_spike_statistics(self.stim_table.index.values,
                        self.unit_ids)

        return self._conditionwise_statistics


    @property
    def presentationwise_statistics(self):
        """ Construct a dataframe with the statistics for each stimulus presentation, by unit

        Returns
        =======
        pd.DataFrame :
            MultiIndex : unit_id, stimulus_presentation_id
            Columns : spike_count, stimulus_condition_id, running_speed 

        """

        if self._presentationwise_statistics is None:

            df = \
                    self.ecephys_session.presentationwise_spike_counts(
                        bin_edges = np.linspace(0, self._trial_duration, 2),
                        stimulus_presentation_ids = self.stim_table.index.values,
                        unit_ids = self.unit_ids,
                    ).to_dataframe().reset_index(level=1, drop=True)

            df = df.join(self.stim_table.loc[df.index.levels[0].values]['stimulus_condition_id'])
            self._presentationwise_statistics = df.join(self.running_speed)

        return self._presentationwise_statistics


    @property
    def stimulus_conditions(self):
        """ Construct a dataframe with the stimulus conditions

        Returns
        =======
        pd.DataFrame :
            Index : stimulus_condition_id
            Columns : stimulus parameter types

        """

        if self._stimulus_conditions is None:

            condition_list = self.stim_table.stimulus_condition_id.unique()

            self._stimulus_conditions = \
                    self.ecephys_session.stimulus_conditions[
                        self.ecephys_session.stimulus_conditions.index.isin(condition_list)
                    ]

        return self._stimulus_conditions


    @property
    def running_speed(self):
        """ Construct a dataframe with the running speed for each trial

        Return
        ======
        pd.DataFrame:
            Index : presentation_id
            Columns : running_speed
        
        """
        if self._running_speed is None:
            
            self._running_speed = pd.DataFrame(index=stim_table.index.values, 
                                               data = {'running_speed' :
                                                    [get_velocity_for_presentation(i) for i in self.stim_table.index.values]
                                                }
                        ).rename_axis('stimulus_presentation_id')

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
    def metrics(self):
        """Returns a pandas DataFrame of the stimulus response metrics for each unit."""
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
        shuffled_mean = np.empty((self.unit_count, n_samples))
        idx = np.random.choice(np.arange(self.stim_table_spontaneous['start_time'].iloc[0],
                                         self.stim_table_spontaneous['stop_time'].iloc[0],
                                         step_size), n_samples)  # TODO: what step size for np.arange?
        for shuf in range(n_samples):
            for i, v in enumerate(self.spikes.keys()):
                spikes = self.spikes[v]
                shuffled_mean[i, shuf] = len(spikes[(spikes > idx[shuf]) & (spikes < (idx[shuf] + offset))])

        sweep_p_values = pd.DataFrame(index=self.stim_table.index.values, columns=self.sweep_events.columns)
        for i, unit_id in enumerate(self.spikes.keys()):
            subset = self.mean_sweep_events[unit_id].values
            null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(subset), 1))
            actual_is_less = subset.reshape(len(subset), 1) <= null_dist_mat
            p_values = np.mean(actual_is_less, axis=1)
            sweep_p_values[unit_id] = p_values

        return sweep_p_values

    def get_velocity_for_presentation(self, presentation_id):

        indices = (self.ecephys_session.running_speed.start_time >= self.stim_table.loc[presentation_id]['start_time']) & \
            (self.ecephys_session.running_speed.start_time < self.stim_table.loc[presentation_id]['stop_time'])
        
        return self.ecephys_session.running_speed[indices]['velocity'].mean()


def get_reliability(unit_sweeps, padding=1.0, num_timestep_second=30, filter_width=0.1, window_beg=0, window_end=None):
    """Computes the trial-to-trial reliability for a set of sweeps for a given cell

    :param unit_sweeps:
    :param padding:
    :return:
    """
    if isinstance(unit_sweeps, (list, tuple)):
        unit_sweeps = np.array([np.array(l) for l in unit_sweeps])

    unit_sweeps = unit_sweeps + padding  # DO NOT use the += as for python arrays that will do in-place modification
    corr_matrix = np.empty((len(unit_sweeps), len(unit_sweeps)))
    fr_window = slice(window_beg, window_end)
    for i in range(len(unit_sweeps)):
        fri = get_fr(unit_sweeps[i], num_timestep_second=num_timestep_second, filter_width=filter_width)
        for j in range(len(unit_sweeps)):
            frj = get_fr(unit_sweeps[j], num_timestep_second=num_timestep_second, filter_width=filter_width)
            # Warning: the pearson coefficient is likely to have a denominator of 0 for some cells/stimulus and give
            # a divide by 0 warning.
            r, p = st.pearsonr(fri[fr_window], frj[fr_window])
            corr_matrix[i, j] = r

    inds = np.triu_indices(len(unit_sweeps), k=1)
    upper = corr_matrix[inds[0], inds[1]]
    return np.nanmean(upper)


def get_fr(spikes, num_timestep_second=30, sweep_length=3.1, filter_width=0.1):
    """Uses a gaussian convolution to convert the spike-times into a contiguous firing-rate series.

    :param spikes: An array of spike times (shifted to start at 0)
    :param num_timestep_second: The sampling frequency
    :param sweep_length: The lenght of the returned array
    :param filter_width: The window of the gaussian method
    :return: A linear-spaced array of length num_timestep_second*sweep_length of the smoothed firing rates series.
    """
    # TODO: figure out the approiate sweep-length from the stimulus
    spikes = spikes.astype(float)
    spike_train = np.zeros((int(sweep_length*num_timestep_second)))
    spike_train[(spikes*num_timestep_second).astype(int)] = 1
    filter_width = int(filter_width*num_timestep_second)
    fr = ndi.gaussian_filter(spike_train, filter_width)
    return fr


def get_lifetime_sparseness(responses):
    """Computes the lifetime sparseness across all the (mean) responses. See Olsen & Wilson 2008.

    :param response_data: A floating-point vector/matrix of dimension N-Responses x M-Cells of the response values
        to the different stimuli.
    :return: An array of size M-Cells that calculates the lifetime sparseness for each cell.
    """
    if responses.ndim == 1:
        # In the (rare) case their is only one cell, turn into a Mx1 matrix the function belows returns a 1x1 array
        # instead of scalar.
        responses = np.array([responses]).T
    coeff = 1.0/responses.shape[0]
    return (1.0 - coeff*((np.power(responses.sum(axis=0), 2)) / (np.power(responses, 2).sum(axis=0)))) / (1.0 - coeff)


def get_osi(responses, ori_vals, in_radians=False):
    """Computes the orientation selectivity of a cell. The calculation of the orientation is done using the normalized
    circular variance (CirVar) as described in Ringbach 2002

    :param tuning: Array of length N. Each value the (averaged) response of the cell at a differenet orientation.
    :param ori_vals: Array of length N. Each value the oriention of the stimulus.
    :param in_radians: Set to True if ori_vals is in units of radians. Default: False
    :return: An N-dimensional array of the circular variance (scalar value, in radians) of the responses.
    """
    # TODO: Try and vectorize function so that it can take in a matrix of N-orientations x M-cells
    ori_rad = ori_vals if in_radians else np.deg2rad(ori_vals)
    num_ori = len(ori_rad)
    cv_top_os = np.empty(num_ori, dtype=np.complex128)
    for i in range(num_ori):
        cv_top_os[i] = (responses[i] * np.exp(1j * 2 * ori_rad[i]))

    return np.abs(cv_top_os.sum()) / responses.sum()


def get_dsi(responses, ori_vals, in_radians=False):
    """Computes the direction selectivity of a cell. See Ringbach 2002, Van Hooser 2014

    :param tuning: Array of length N. Each value the (averaged) response of the cell at a differenet orientation.
    :param ori_vals: Array of length N. Each value the oriention of the stimulus.
    :param in_radians: Set to True if ori_vals is in units of radians. Default: False
    :return: An N-dimensional array of the circular variance (scalar value, in radians) of the responses.
    """
    # TODO: Try and vectorize function so that it can take in a matrix of N-orientations x M-cells
    ori_rad = ori_vals if in_radians else np.deg2rad(ori_vals)
    num_ori = len(ori_rad)
    cv_top_ds = np.empty(num_ori, dtype=np.complex128)
    for i in range(num_ori):
        cv_top_ds[i] = (responses[i] * np.exp(1j * ori_rad[i]))

    return np.abs(cv_top_ds.sum()) / responses.sum()


def get_running_modulation(mean_sweep_runs, mean_sweep_stats):
    """computes running modulation of cell at its preferred condition provided there are at least 2 trials for both
    stationary and running conditions

    :param mean_sweep_runs:
    :param mean_sweep_stats:
    :return: p_value of running modulation, running modulation metric, mean response to preferred condition when
    running mean response to preferred condition when stationary
    """
    if np.logical_and(len(mean_sweep_runs) > 1, len(mean_sweep_stats) > 1):
        run = mean_sweep_runs.mean()
        stat = mean_sweep_stats.mean()
        if run > stat:
            run_mod = (run - stat) / run
        elif stat > run:
            run_mod = -1 * (stat - run) / stat
        else:
            run_mod = 0
        (_, p) = st.ttest_ind(mean_sweep_runs, mean_sweep_stats, equal_var=False)
        return p, run_mod, run, stat
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN