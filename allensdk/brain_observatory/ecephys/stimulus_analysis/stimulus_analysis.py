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
        self._presentationwise_statistics = None
        self._presentationwise_spikes = None
        self._conditionwise_psth = None
        self._stimulus_conditions = None

        self._spikes = None
        self._stim_table_spontaneous = None
        self._stimulus_names = None
        self._running_speed = None
        self._sweep_events = None
        self._mean_sweep_events = None
        self._sweep_p_values = None
        self._metrics = None

        self._psth_resolution = 0.01 # ms

        self._trial_duration = None
        self._preferred_condition = {}


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
    def name(self):
        """ Return the stimulus name."""
        return self._module_name

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
    def stim_table(self):
        # Stimulus table is already in EcephysSession object, just need to subselect presentations for this stimulus.
        if self._stim_table is None:
            # TODO: Give warning if no stimulus
            if self._stimulus_names is None:
                # Older versions of NWB files the stimulus name is in the form stimulus_gratings_N, so if
                # self._stimulus_names is not explicity specified try to figure out stimulus
                stims_table = self.ecephys_session.stimulus_presentations
                #print(stims_table['stimulus_name'].unique())
                #print(self._stimulus_key)
                stim_names = [s for s in stims_table['stimulus_name'].unique()
                              if s.startswith(self._stimulus_key)]

                self._stim_table = stims_table[stims_table['stimulus_name'].isin(stim_names)]
                #print(stim_names)
            else:
                self._stimulus_names = [self._stimulus_names] if isinstance(self._stimulus_names, string_types) \
                    else self._stimulus_names
                self._stim_table = self.ecephys_session.get_presentations_for_stimulus(self._stimulus_names)

        return self._stim_table

    @property
    def metrics_names(self):
        return [c[0] for c in self.METRICS_COLUMNS]

    @property
    def metrics_dtypes(self):
        return [c[1] for c in self.METRICS_COLUMNS]

    @property
    def METRICS_COLUMNS(self):
        raise NotImplementedError

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
    def null_condition(self):
        raise NotImplementedError()


    @property
    def conditionwise_psth(self):
        """ Construct a PSTH for each stimulus condition, by unit

        Returns
        =======
        xarray.DataArray :
            Coordinates: 
                - stimulus_condition_id
                - time_relative_to_stimulus_onset
                - unit_id
        """

        if self._conditionwise_psth is None:

            dataset = self.ecephys_session.presentationwise_spike_counts(
                bin_edges = np.arange(0,self._trial_duration ,self._psth_resolution),
                stimulus_presentation_ids = self.stim_table.index.values,
                unit_ids = self.unit_ids
                )

            da = dataset['spike_counts'].assign_coords(
                        stimulus_presentation_id=self.stim_table['stimulus_condition_id'].values)
            da = da.rename({'stimulus_presentation_id': 'stimulus_condition_id'})

            self._conditionwise_psth = da.groupby('stimulus_condition_id').mean(dim='stimulus_condition_id')

        return self._conditionwise_psth


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
    def presentationwise_spike_times(self):
        """ Construct a dataframe with the spike times for each stimulus presentation

        Returns
        =======
        pd.DataFrame :
            Index : spike time
            Columns : stimulus_presentation_id, unit_id 

        """

        if self._presentationwise_spikes is None:

            self._presentationwise_spikes = self.ecephys_session.presentationwise_spike_times(
                    stimulus_presentation_ids = self.stim_table.index.values,
                    unit_ids = self.unit_ids,
                )

        return self._presentationwise_statistics



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

            df = self.ecephys_session.presentationwise_spike_counts(
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
            
            self._running_speed = pd.DataFrame(index=self.stim_table.index.values, 
                                               data = {'running_speed' :
                                                    [self.get_velocity_for_presentation(i) for i in self.stim_table.index.values]
                                                }
                        ).rename_axis('stimulus_presentation_id')

        return self._running_speed

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

    def get_running_modulation(self, unit_id, preferred_condition, threshold=1):
        """Computes running modulation of a unit at its preferred condition provided there are at least 2 trials for both
        stationary and running conditions

        :param pref_ori:
        :param pref_tf:
        :param v:
        :return: p_value of running modulation, mean response to preferred condition when running, mean response to
        preferred condition when stationary
        """

        subset = self.presentationwise_statistics[
                    self.presentationwise_statistics['stimulus_condition_id'] == preferred_condition
                    ].xs(unit_id, level=1)

        spike_counts = subset['spike_counts'].values 
        running_speeds = subset['running_speed'].values 

        is_running = running_speeds >= threshold

        if 1 < np.sum(is_running) < (len(running_speeds) - 1):

            run = spike_counts[is_running]
            stat = spike_counts[np.invert(is_running)]

            run_mean = np.mean(run)
            stat_mean = np.mean(stat)

            if run_mean > stat_mean:
                run_mod = (run_mean - stat_mean) / run_mean
            else:
                run_mod = -1 * (stat_mean - run_mean) / stat_mean
            (_, p) = st.ttest_ind(run, stat, equal_var=False)
            return p, run_mod
        else:
            return np.NaN, np.NaN

    def get_preferred_condition(self, unit_id):

        if unit_id not in self._preferred_condition:

            try:
                df = self.conditionwise_statistics.drop(index=self.null_condition, level=1)
            except IndexError:
                df = self.conditionwise_statistics

            self._preferred_condition[unit_id] = df.loc[unit_id]['spike_mean'].idxmax()

        return self._preferred_condition[unit_id]

    def empty_metrics_table(self):
        # pandas can have issues interpreting type and makes the column 'object' type, this should enforce the
        # correct data type for each column
        return pd.DataFrame(np.empty(self.unit_count, dtype=np.dtype(self.METRICS_COLUMNS)),
                                   index=self.unit_ids).rename_axis('unit_id')


    def get_lifetime_sparseness(self, unit_id):
        """Computes lifetime sparseness of responses for one unit
        :return:
        """
        df = self.conditionwise_statistics.drop(index=self.null_condition, level=1)
        responses = df.loc[unit_id]['spike_count'].values 

        return lifetime_sparseness(responses)

    def get_fano_factor(self, unit_id, preferred_condition):

        # Fano factor calculation goes here:
        #   Equal to variance of spike rate divided by the mean spike rate
        #   See: https://en.wikipedia.org/wiki/Fano_factor

        return np.nan


    def get_time_to_peak(self, unit_id, preferred_condition):

        # Time-to-peak calculation goes here:
        #   Equal to the time of the maximum firing rate of the average PSTH at the preferred condition

        return np.nan

    def get_reliability(self, unit_id, preferred_condition):

        # Reliability calculation goes here:
        #   Depends on the trial-to-trial correlation of the smoothed response
        #   What smoothing window is appropriate for ephys? We need to test this more

        return np.nan

    def get_overall_firing_rate(self, unit_id):

        # Firing rate calculation goes here:
        #   This is the average firing rate over the entire stimulus interval

        return np.nan


def reliability(unit_sweeps, padding=1.0, num_timestep_second=30, filter_width=0.1, window_beg=0, window_end=None):
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


def lifetime_sparseness(responses):
    """Computes the lifetime sparseness for one unit. See Olsen & Wilson 2008.

    :param responses: A floating-point vector of N responses for one unit
    :return: The lifetime sparseness for one unit

    ### CHECK THIS CALCULATION!
    """

    coeff = 1.0/len(responses)

    return (1.0 - coeff*((np.power(np.sum(responses), 2)) / (np.sum(np.power(responses, 2))))) / (1.0 - coeff)


def osi(orivals, tuning):
    """Computes orientation selectivity for a tuning curve 

    """

    cv_top = tuning * np.exp(1j * 2 * orivals)
    return np.abs(cv_top.sum()) / tuning.sum()


def dsi(orivals, tuning):
    """Computes direction selectivity for a tuning curve 

    """

    cv_top = tuning * np.exp(1j * orivals)
    return np.abs(cv_top.sum()) / tuning.sum()

def deg2rad(arr):

    """ Converts array-like input from degrees to radians
    """

    return arr / 180 * np.pi 