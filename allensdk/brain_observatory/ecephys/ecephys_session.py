import warnings
from collections.abc import Collection
from collections import defaultdict

import xarray as xr
import numpy as np
import pandas as pd
import scipy.stats

from allensdk.core.lazy_property import LazyPropertyMixin
from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi, EcephysNwbSessionApi, EcephysNwb1Api
from allensdk.brain_observatory.ecephys.stimulus_table import naming_utilities
from allensdk.brain_observatory.ecephys.stimulus_table._schemas import default_stimulus_renames, default_column_renames    


NON_STIMULUS_PARAMETERS = tuple([
    'start_time',
    'stop_time',
    'duration',
    'stimulus_block',
    "stimulus_condition_id"
]) # stimulus_presentation column names not describing a parameter of a stimulus


class EcephysSession(LazyPropertyMixin):
    ''' Represents data from a single EcephysSession

    Attributes
    ----------
    units : pd.Dataframe
        A table whose rows are sorted units (putative neurons) and whose columns are characteristics 
        of those units.
        Index is:
            unit_id : int
                Unique integer identifier for this unit.
        Columns are:
            firing_rate : float
                This unit's firing rate (spikes / s) calculated over the window of that unit's activity 
                (the time from its first detected spike to its last).
            isi_violations : float
                Estamate of this unit's contamination rate (larger means that more of the spikes assigned 
                to this unit probably originated from other neurons). Calculated as a ratio of the firing 
                rate of the unit over periods where spikes would be isi-violating vs the total firing 
                rate of the unit.
            peak_channel_id : int
                Unique integer identifier for this unit's peak channel. A unit's peak channel is the channel on 
                which its peak-to-trough amplitude difference is maximized. This is assessed using the kilosort 2 
                templates rather than the mean waveforms for a unit.
            snr : float
                Signal to noise ratio for this unit.
            probe_horizontal_position :  numeric
                The horizontal (short-axis) position of this unit's peak channel in microns.
            probe_vertical_position : numeric
                The vertical (long-axis, lower values are closer to the probe base) position of 
                this unit's peak channel in microns.
            probe_id : int
                Unique integer identifier for this unit's probe.
            probe_description : str
                Human-readable description carrying miscellaneous information about this unit's probe.
            location : str
                Gross-scale location of this unit's probe.
    spike_times : dict
        Maps integer unit ids to arrays of spike times (float) for those units.
    running_speed : RunningSpeed
        NamedTuple with two fields
            timestamps : numpy.ndarray
                Timestamps of running speed data samples
            values : np.ndarray
                Running speed of the experimental subject (in cm / s).
    mean_waveforms : dict
        Maps integer unit ids to xarray.DataArrays containing mean spike waveforms for that unit.
    stimulus_presentations : pd.DataFrame
        Table whose rows are stimulus presentations and whose columns are presentation characteristics. 
        A stimulus presentation is the smallest unit of distinct stimulus presentation and lasts for 
        (usually) 1 60hz frame. Since not all parameters are relevant to all stimuli, this table 
        contains many 'null' values.
        Index is
            stimulus_presentation_id : int
                Unique identifier for this stimulus presentation
        Columns are
            start_time :  float
                Time (s) at which this presentation began
            stop_time : float
                Time (s) at which this presentation ended
            duration : float
                stop_time - start_time (s). Included for convenience.
            stimulus_name : str
                Identifies the stimulus family (e.g. "drifting_gratings" or "natural_movie_3") used 
                for this presentation. The stimulus family, along with relevant parameter values, provides the 
                information required to reconstruct the stimulus presented during this presentation. The empty 
                string indicates a blank period.
            stimulus_block : numeric
                A stimulus block is made by sequentially presenting presentations from the same stimulus family. 
                This value is the index of the block which contains this presentation. During a blank period, 
                this is 'null'.
            TF : float
                Temporal frequency, or 'null' when not appropriate.
            SF : float
                Spatial frequency, or 'null' when not appropriate
            Ori : float
                Orientation (in degrees) or 'null' when not appropriate
            Contrast : float
            Pos_x : float
            Pos_y : float
            Color : numeric
            Image : numeric
            Phase : float
            stimulus_condition_id : integer
                identifies the session-unique stimulus condition (permutation of parameters) to which this presentation 
                belongs
    stimulus_conditions : pd.DataFrame
        Each row is a unique permutation (within this session) of stimulus parameters presented during this experiment. 
        Columns are as stimulus presentations, sans start_time, end_time, stimulus_block, and duration.
    inter_presentation_intervals : pd.DataFrame
        The elapsed time between each immediately sequential pair of stimulus presentations. This is a dataframe with a 
        two-level multiindex (levels are 'from_presentation_id' and 'to_presentation_id'). It has a single column, 
        'interval', which reports the elapsed time between the two presentations in seconds on the experiment's master 
        clock.

    '''


    @property
    def num_units(self):
        return self.units.shape[0]


    @property
    def num_probes(self):
        return self.probes.shape[0]


    @property
    def num_channels(self):
        return self.channels.shape[0]


    @property
    def num_stimulus_presentations(self):
        return self.stimulus_presentations.shape[0]


    @property
    def stimulus_names(self):
        return self.stimulus_presentations['stimulus_name'].unique().tolist()


    @property
    def stimulus_conditions(self):
        self.stimulus_presentations
        return self._stimulus_conditions


    @property
    def corneal_reflection_ellipse_fits(self):
        return self._eye_tracking_ellipse_fit_data["cr_ellipse_fits"]


    @property
    def eye_ellipse_fits(self):
        return self._eye_tracking_ellipse_fit_data["eye_ellipse_fits"]


    @property
    def pupil_ellipse_fits(self):
        return self._eye_tracking_ellipse_fit_data["pupil_ellipse_fits"]
    

    @property
    def rig_geometry_data(self):
        return self._eye_tracking_ellipse_fit_data["rig_geometry_data"]


    @property
    def rig_equipment_name(self):
        return self._eye_tracking_ellipse_fit_data["rig_equipment"]


    @property
    def specimen_name(self):
        return self._metadata["specimen_name"]


    @property
    def age_in_days(self):
        return self._metadata["age_in_days"]


    @property
    def sex(self):
        return self._metadata["sex"]


    @property
    def full_genotype(self):
        return self._metadata["full_genotype"]


    @property
    def session_type(self):
        return self._metadata["stimulus_name"]


    def __init__(self, api, **kwargs):
        self.api: EcephysSessionApi = api

        self.ecephys_session_id = self.LazyProperty(self.api.get_ecephys_session_id)
        self.session_start_time = self.LazyProperty(self.api.get_session_start_time)
        self.running_speed= self.LazyProperty(self.api.get_running_speed)
        self.mean_waveforms = self.LazyProperty(self.api.get_mean_waveforms, wrappers=[self._build_mean_waveforms])
        self.spike_times = self.LazyProperty(self.api.get_spike_times, wrappers=[self._build_spike_times])
        self.optogenetic_stimulation_epochs = self.LazyProperty(self.api.get_optogenetic_stimulation)
        self.spike_amplitudes = self.LazyProperty(self.api.get_spike_amplitudes)

        self.probes = self.LazyProperty(self.api.get_probes)
        self.channels = self.LazyProperty(self.api.get_channels)

        self.stimulus_presentations = self.LazyProperty(self.api.get_stimulus_presentations, wrappers=[self._build_stimulus_presentations])
        self.units = self.LazyProperty(self.api.get_units, wrappers=[self._build_units_table])
        self.inter_presentation_intervals = self.LazyProperty(self._build_inter_presentation_intervals)
        self.invalid_times = self.LazyProperty(self.api.get_invalid_times)

        self._eye_tracking_ellipse_fit_data = self.LazyProperty(self.api.get_eye_tracking_ellipse_fit_data)
        self.raw_eye_gaze_mapping_data = self.LazyProperty(self.api.get_raw_eye_gaze_mapping_data)
        self.filtered_eye_gaze_mapping_data = self.LazyProperty(self.api.get_filtered_eye_gaze_mapping_data)

        self._metadata = self.LazyProperty(self.api.get_metadata)


    def get_current_source_density(self, probe_id):
        """ Obtain current source density (CSD) of trial-averaged response to a flash stimuli for this probe.
        See allensdk.brain_observatory.ecephys.current_source_density for details of CSD calculation.

        CSD is computed with a 1D method (second spatial derivative) without prior spatial smoothing
        User should apply spatial smoothing of their choice (e.g., Gaussian filter) to the computed CSD


        Parameters
        ----------
        probe_id : int
            identify the probe whose CSD data ought to be loaded

        Returns
        -------
        xr.DataArray :
            dimensions are channel (id) and time (seconds, relative to stimulus onset). Values are current source 
            density assessed on that channel at that time (V/m^2)

        """

        return self.api.get_current_source_density(probe_id)


    def get_lfp(self, probe_id):
        ''' Load an xarray DataArray with LFP data from channels on a single probe

        Parameters
        ----------
        probe_id : int
            identify the probe whose LFP data ought to be loaded

        Returns
        -------
        xr.DataArray :
            dimensions are channel (id) and time (seconds). Values are sampled LFP data.

        Notes
        -----
        Unlike many other data access methods on this class. This one does not cache the loaded data in memory due to 
        the large size of the LFP data.

        '''

        return self.api.get_lfp(probe_id)


    def get_inter_presentation_intervals_for_stimulus(self, stimulus_names):
        ''' Get a subset of this session's inter-presentation intervals, filtered by stimulus name.

        Parameters
        ----------
        stimulus_names : array-like of str
            The names of stimuli to include in the output.

        Returns
        -------
        pd.DataFrame : 
            inter-presentation intervals, filtered to the requested stimulus names.

        '''

        stimulus_names = warn_on_scalar(stimulus_names, f'expected stimulus_names to be a collection (list-like), but found {type(stimulus_names)}: {stimulus_names}')
        filtered_presentations = self.stimulus_presentations[self.stimulus_presentations['stimulus_name'].isin(stimulus_names)]
        filtered_ids = set(filtered_presentations.index.values)

        return self.inter_presentation_intervals[
            (self.inter_presentation_intervals.index.isin(filtered_ids, level='from_presentation_id'))
            & (self.inter_presentation_intervals.index.isin(filtered_ids, level='to_presentation_id'))
        ]


    def get_presentations_for_stimulus(self, stimulus_names):
        '''Get a subset of stimulus presentations by name, with irrelevant parameters filtered off

        Parameters
        ----------
        stimulus_names : array-like of str
            The names of stimuli to include in the output.

        Returns
        -------
        pd.DataFrame :
            Rows are filtered presentations, columns are the relevant subset of stimulus parameters

        '''

        stimulus_names = warn_on_scalar(stimulus_names, f'expected stimulus_names to be a collection (list-like), but found {type(stimulus_names)}: {stimulus_names}')
        filtered_presentations = self.stimulus_presentations[self.stimulus_presentations['stimulus_name'].isin(stimulus_names)]
        return removed_unused_stimulus_presentation_columns(filtered_presentations)


    def get_stimulus_epochs(self, duration_thresholds=None):
        """ Reports continuous periods of time during which a single kind of stimulus was presented

        Parameters
        ---------
        duration_thresholds : dict, optional
            keys are stimulus names, values are floating point durations in seconds. All epochs with
                - a given stimulus name
                - a duration shorter than the associated threshold
            will be removed from the results

        """

        if duration_thresholds is None:
            duration_thresholds = {"spontaneous_activity": 90.0}

        presentations = self.stimulus_presentations.copy()
        diff_indices = nan_intervals(presentations["stimulus_block"].values)

        epochs = []
        for left, right in zip(diff_indices[:-1], diff_indices[1:]):
            epochs.append({
                "start_time": presentations.iloc[left]["start_time"],
                "stop_time": presentations.iloc[right-1]["stop_time"],
                "stimulus_name": presentations.iloc[left]["stimulus_name"],
                "stimulus_block": presentations.iloc[left]["stimulus_block"]
            })
        epochs = pd.DataFrame(epochs)
        epochs["duration"] = epochs["stop_time"] - epochs["start_time"]

        for key, threshold in duration_thresholds.items():
            epochs = epochs[
                (epochs["stimulus_name"] != key)
                | (epochs["duration"] >= threshold)
            ]

        return epochs.loc[:, ["start_time", "stop_time", "duration", "stimulus_name", "stimulus_block"]]

    def get_invalid_times(self):
        """ Report invalid time intervals with tags describing the scope of invalid data

        The tags format: [scope,scope_id,label]

        scope:
            'EcephysSession': data is invalid across session
            'EcephysProbe': data is invalid for a single probe
        label:
            'all_probes': gain fluctuations on the Neuropixels probe result in missed spikes and LFP saturation events
            'stimulus' : very long frames (>3x the normal frame length) make any stimulus-locked analysis invalid
            'probe#': probe # stopped sending data during this interval (spikes and LFP samples will be missing)
            'optotagging': missing optotagging data

        Returns
        -------
        pd.DataFrame :
            Rows are invalid intervals, columns are 'start_time' (s), 'stop_time' (s), 'tags'
        """

        return self.invalid_times


    def presentationwise_spike_counts(
        self, 
        bin_edges, 
        stimulus_presentation_ids, 
        unit_ids, 
        binarize=False, 
        dtype=None, 
        large_bin_size_threshold=0.001,
        time_domain_callback=None
    ):
        ''' Build an array of spike counts surrounding stimulus onset per unit and stimulus frame.

        Parameters
        ---------
        bin_edges : numpy.ndarray
            Spikes will be counted into the bins defined by these edges. Values are in seconds, relative 
            to stimulus onset.
        stimulus_presentation_ids : array-like
            Filter to these stimulus presentations
        unit_ids : array-like
            Filter to these units
        binarize : bool, optional
            If true, all counts greater than 0 will be treated as 1. This results in lower storage overhead, 
            but is only reasonable if bin sizes are fine (<= 1 millisecond).
        large_bin_size_threshold : float, optional
            If binarize is True and the largest bin width is greater than this value, a warning will be emitted.
        time_domain_callback : callable, optional
            The time domain is a numpy array whose values are trial-aligned bin 
            edges (each row is aligned to a different trial). This optional function will be 
            applied to the time domain before counting spikes.

        Returns
        -------
        xarray.DataArray :
            Data array whose dimensions are stimulus presentation, unit, 
            and time bin and whose values are spike counts.

        '''

        stimulus_presentations = self._filter_owned_df('stimulus_presentations', ids=stimulus_presentation_ids)
        units = self._filter_owned_df('units', ids=unit_ids)

        largest_bin_size = np.amax(np.diff(bin_edges))
        if binarize and largest_bin_size > large_bin_size_threshold:
            warnings.warn(
                f'You\'ve elected to binarize spike counts, but your maximum bin width is {largest_bin_size:2.5f} seconds. '
                'Binarizing spike counts with such a large bin width can cause significant loss of accuracy! '
                f'Please consider only binarizing spike counts when your bins are <= {large_bin_size_threshold} seconds wide.'
            )

        bin_edges = np.array(bin_edges)
        domain = build_time_window_domain(bin_edges, stimulus_presentations['start_time'].values, callback=time_domain_callback)

        out_of_order = np.where(np.diff(domain, axis=1) < 0)
        if len(out_of_order[0]) > 0:
            out_of_order_time_bins = [(row, col) for row, col in zip(out_of_order)]
            raise ValueError(f"The time domain specified contains out-of-order bin edges at indices: {out_of_order_time_bins}")

        ends = domain[:, -1]
        starts = domain[:, 0]
        time_diffs = starts[1:] - ends[:-1]
        overlapping = np.where(time_diffs < 0)[0]

        if len(overlapping) > 0:
            # Ignoring intervals that overlaps multiple time bins because trying to figure that out would take O(n)
            overlapping = [(s, s+1) for s in overlapping]
            warnings.warn(f"You've specified some overlapping time intervals between neighboring rows: {overlapping}, "
                          f"with a maximum overlap of {np.abs(np.min(time_diffs))} seconds.")

        tiled_data = build_spike_histogram(
            domain, self.spike_times, units.index.values, dtype=dtype, binarize=binarize
        )

        tiled_data = xr.DataArray(
            name='spike_counts',
            data=tiled_data, 
            coords={
                'stimulus_presentation_id': stimulus_presentations.index.values,
                'time_relative_to_stimulus_onset': bin_edges[:-1] + np.diff(bin_edges) / 2,
                'unit_id': units.index.values
            },
            dims=['stimulus_presentation_id', 'time_relative_to_stimulus_onset', 'unit_id']
        )

        return tiled_data


    def presentationwise_spike_times(self, stimulus_presentation_ids=None, unit_ids=None):
        ''' Produce a table associating spike times with units and stimulus presentations

        Parameters
        ----------
        stimulus_presentation_ids : array-like
            Filter to these stimulus presentations
        unit_ids : array-like
            Filter to these units

        Returns
        -------
        pandas.DataFrame : 
        Index is
            spike_time : float
                On the session's master clock.
        Columns are
            stimulus_presentation_id : int
                The stimulus presentation on which this spike occurred.
            unit_id : int
                The unit that emitted this spike.
        '''

        stimulus_presentations = self._filter_owned_df('stimulus_presentations', ids=stimulus_presentation_ids)
        units = self._filter_owned_df('units', ids=unit_ids)

        presentation_times = np.zeros([stimulus_presentations.shape[0] * 2])
        presentation_times[::2] = np.array(stimulus_presentations['start_time'])
        presentation_times[1::2] = np.array(stimulus_presentations['stop_time'])
        all_presentation_ids = np.array(stimulus_presentations.index.values)

        presentation_ids = []
        unit_ids = []
        spike_times = []

        for ii, unit_id in enumerate(units.index.values):
            data = self.spike_times[unit_id]
            indices = np.searchsorted(presentation_times, data) - 1

            index_valid = indices % 2 == 0
            presentations = all_presentation_ids[np.floor(indices / 2).astype(int)]

            sorder = np.argsort(presentations)
            presentations = presentations[sorder]
            index_valid = index_valid[sorder]
            data = data[sorder]

            changes = np.where(np.ediff1d(presentations, to_begin=1, to_end=1))[0]
            for ii, jj in zip(changes[:-1], changes[1:]):
                values = data[ii:jj][index_valid[ii:jj]]
                if values.size == 0:
                    continue

                unit_ids.append(np.zeros([values.size]) + unit_id)
                presentation_ids.append(np.zeros([values.size]) + presentations[ii])
                spike_times.append(values)

        if not spike_times:
            # If there are no units firing during the given stimulus return an empty dataframe
            return pd.DataFrame(columns=['spike_times', 'stimulus_presentation', 'unit_id'])

        return pd.DataFrame({
            'stimulus_presentation_id': np.concatenate(presentation_ids).astype(int),
            'unit_id': np.concatenate(unit_ids).astype(int)
        }, index=pd.Index(np.concatenate(spike_times), name='spike_time')).sort_values('spike_time', axis=0)


    def conditionwise_spike_statistics(self, stimulus_presentation_ids=None, unit_ids=None):
        """ Produce summary statistics for each distinct stimulus condition

        Parameters
        ----------
        stimulus_presentation_ids : array-like
            identifies stimulus presentations from which spikes will be considered
        unit_ids : array-like
            identifies units whose spikes will be considered

        Returns
        -------
        pd.DataFrame :
            Rows are indexed by unit id and stimulus condition id. Values are summary statistics describing spikes 
            emitted by a specific unit across presentations within a specific condition.

        """
        # TODO: Need to return an empty df if no matching unit-ids or presentation-ids are found
        # TODO: To use filter_owned_df() make sure to convert the results from a Series to a Dataframe
        stimulus_presentation_ids = stimulus_presentation_ids if stimulus_presentation_ids is not None else \
                self.stimulus_presentations.index.values  # In case
        presentations = self.stimulus_presentations.loc[stimulus_presentation_ids, ["stimulus_condition_id"]]

        spikes = self.presentationwise_spike_times(
            stimulus_presentation_ids=stimulus_presentation_ids, unit_ids=unit_ids
        )

        if spikes.empty:
            # In the case there are no spikes
            spike_counts = pd.DataFrame({'spike_count': 0},
                                        index=pd.MultiIndex.from_product([stimulus_presentation_ids, unit_ids],
                                                                         names=['stimulus_presentation_id', 'unit_id']))

        else:
            spike_counts = spikes.copy()
            spike_counts["spike_count"] = np.zeros(spike_counts.shape[0])
            spike_counts = spike_counts.groupby(["stimulus_presentation_id", "unit_id"]).count()
            unit_ids = unit_ids if unit_ids is not None else spikes['unit_id'].unique()  # If not explicity stated get unit ids from spikes table.
            spike_counts = spike_counts.reindex(pd.MultiIndex.from_product([stimulus_presentation_ids,
                                                                            unit_ids],
                                                                           names=['stimulus_presentation_id',
                                                                                  'unit_id']), fill_value=0)

        sp = pd.merge(spike_counts, presentations, left_on="stimulus_presentation_id", right_index=True, how="left")
        sp.reset_index(inplace=True)

        summary = []
        for ind, gr in sp.groupby(["stimulus_condition_id", "unit_id"]):
            summary.append({
                "stimulus_condition_id": ind[0],
                "unit_id": ind[1],
                "spike_count": gr["spike_count"].sum(),
                "stimulus_presentation_count": gr.shape[0],
                "spike_mean": np.mean(gr["spike_count"].values),
                "spike_std": np.std(gr["spike_count"].values, ddof=1),
                "spike_sem": scipy.stats.sem(gr["spike_count"].values)
            })

        return pd.DataFrame(summary).set_index(keys=["unit_id", "stimulus_condition_id"])


    def get_parameter_values_for_stimulus(self, stimulus_name, drop_nulls=True):
        """ For each stimulus parameter, report the unique values taken on by that 
        parameter while a named stimulus was presented.

        Parameters
        ----------
        stimulus_name : str
            filter to presentations of this stimulus

        Returns
        -------
        dict : 
            maps parameters (column names) to their unique values.

        """

        presentation_ids = self.get_presentations_for_stimulus([stimulus_name]).index.values
        return self.get_stimulus_parameter_values(presentation_ids, drop_nulls=drop_nulls)


    def get_stimulus_parameter_values(self, stimulus_presentation_ids=None, drop_nulls=True):
        ''' For each stimulus parameter, report the unique values taken on by that 
        parameter throughout the course of the  session.

        Parameters
        ----------
        stimulus_presentation_ids : array-like, optional
            If provided, only parameter values from these stimulus presentations will be considered.

        Returns
        -------
        dict : 
            maps parameters (column names) to their unique values.

        '''

        stimulus_presentations = self._filter_owned_df('stimulus_presentations', ids=stimulus_presentation_ids)
        stimulus_presentations = stimulus_presentations.drop(columns=list(NON_STIMULUS_PARAMETERS) + ['stimulus_name'])
        stimulus_presentations = removed_unused_stimulus_presentation_columns(stimulus_presentations)

        parameters = {}
        for colname in stimulus_presentations.columns:
            uniques = stimulus_presentations[colname].unique()

            non_null = np.array(uniques[uniques != "null"])
            non_null = non_null
            non_null = np.sort(non_null)

            if not drop_nulls and "null" in uniques:
                non_null = np.concatenate([non_null, ["null"]])

            parameters[colname] = non_null

        return parameters

    def channel_structure_intervals(self, channel_ids):

        """ find on a list of channels the intervals of channels inserted into particular structures

        Parameters
        ----------
        channel_ids : list
            A list of channel ids
        structure_id_key : str
            use this column for numerically identifying structures
        structure_label_key : str
            use this column for human-readable structure identification

        Returns
        -------
        labels : np.ndarray
            for each detected interval, the label associated with that interval
        intervals : np.ndarray
            one element longer than labels. Start and end indices for intervals.

        """
        structure_id_key = "structure_id"
        structure_label_key = "structure_acronym"
        np.array(channel_ids).sort()
        table = self.channels.loc[channel_ids]

        unique_probes = table["probe_id"].unique()
        if len(unique_probes)>1:
            warnings.warn("Calculating structure boundaries across channels from multiple probes.")

        intervals = nan_intervals(table[structure_id_key].values)
        labels = table[structure_label_key].iloc[intervals[:-1]].values

        return labels, intervals


    def _build_spike_times(self, spike_times):
        retained_units = set(self.units.index.values)
        output_spike_times = {}

        for unit_id in list(spike_times.keys()):
            data = spike_times.pop(unit_id)
            if unit_id not in retained_units:
                continue
            output_spike_times[unit_id] = data

        return output_spike_times


    def _build_stimulus_presentations(self, stimulus_presentations):
        stimulus_presentations.index.name = 'stimulus_presentation_id'
        stimulus_presentations = stimulus_presentations.drop(columns=['stimulus_index'])

        # TODO: putting these here for now; after SWDB 2019, will rerun stimulus table module for all sessions 
        # and can remove these
        stimulus_presentations = naming_utilities.collapse_columns(stimulus_presentations)
        stimulus_presentations = naming_utilities.standardize_movie_numbers(stimulus_presentations)
        stimulus_presentations = naming_utilities.add_number_to_shuffled_movie(stimulus_presentations)
        stimulus_presentations = naming_utilities.map_stimulus_names(
            stimulus_presentations, default_stimulus_renames
        )
        stimulus_presentations = naming_utilities.map_column_names(stimulus_presentations, default_column_renames)

        # pandas groupby ops ignore nans, so we need a new null value that pandas does not recognize as null ...
        stimulus_presentations[stimulus_presentations == ''] = np.nan
        stimulus_presentations = stimulus_presentations.fillna('null') # 123 / 2**8

        stimulus_presentations['duration'] = stimulus_presentations['stop_time'] - stimulus_presentations['start_time']

        # TODO: database these
        stimulus_conditions = {}
        presentation_conditions = []
        cid_counter = -1

        # TODO: Can we have parameters on what columns to omit? If stimulus_block or duration is left in it can affect
        #   how conditionwise_spike_statistics counts spikes
        params_only = stimulus_presentations.drop(columns=["start_time", "stop_time", "duration", "stimulus_block"])
        for row in params_only.itertuples(index=False):

            if row in stimulus_conditions:
                cid = stimulus_conditions[row]
            else:
                cid_counter += 1
                stimulus_conditions[row] = cid_counter
                cid = cid_counter

            presentation_conditions.append(cid)

        cond_ids = []
        cond_vals = []

        for cv, ci in stimulus_conditions.items():
            cond_ids.append(ci)
            cond_vals.append(cv)

        self._stimulus_conditions = pd.DataFrame(cond_vals, index=pd.Index(data=cond_ids, name="stimulus_condition_id"))
        stimulus_presentations["stimulus_condition_id"] = presentation_conditions

        return stimulus_presentations


    def _build_units_table(self, units_table):
        channels = self.channels.copy()
        probes = self.probes.copy()

        self._unmerged_units = units_table.copy()
        table = pd.merge(units_table, channels, left_on='peak_channel_id', right_index=True, suffixes=['_unit', '_channel'])
        table = pd.merge(table, probes, left_on='probe_id', right_index=True, suffixes=['_unit', '_probe'])

        table.index.name = 'unit_id'
        table = table.rename(columns={
            'description': 'probe_description',
            #'manual_structure_id': 'structure_id',
            #'manual_structure_acronym': 'structure_acronym',
            'local_index_channel': 'channel_local_index',
        })

        return table.sort_values(by=['probe_description', 'probe_vertical_position', 'probe_horizontal_position'])


    def _build_nwb1_waveforms(self, mean_waveforms):
        # _build_mean_waveforms() assumes every unit has the same number of waveforms and that a unit-waveform exists
        # for all channels. This is not true for NWB 1 files where each unit has ONE waveform on ONE channel
        units_df = self.units
        output_waveforms = {}
        sampling_rate_lu = {uid: self.probes.loc[r['probe_id']]['sampling_rate'] for uid, r in units_df.iterrows()}

        for uid in list(mean_waveforms.keys()):
            data = mean_waveforms.pop(uid)
            output_waveforms[uid] = xr.DataArray(
                data=data,
                dims=['channel_id', 'time'],
                coords={
                    'channel_id': [units_df.loc[uid]['peak_channel_id']],
                    'time': np.arange(data.shape[1]) / sampling_rate_lu[uid]
                }
            )

        return output_waveforms

    def _build_mean_waveforms(self, mean_waveforms):
        if isinstance(self.api, EcephysNwb1Api):
            return self._build_nwb1_waveforms(mean_waveforms)

        channel_id_lut = defaultdict(lambda: -1)
        for cid, row in self.channels.iterrows():
            channel_id_lut[(row["local_index"], row["probe_id"])] = cid

        probe_id_lut = {uid: row['probe_id'] for uid, row in self.units.iterrows()}

        output_waveforms = {}
        for uid in list(mean_waveforms.keys()):
            data = mean_waveforms.pop(uid)

            if uid not in probe_id_lut: # It's been filtered out during unit table generation!
                continue

            probe_id = probe_id_lut[uid]
            output_waveforms[uid] = xr.DataArray(
                data=data,
                dims=['channel_id', 'time'],
                coords={
                    'channel_id': [ channel_id_lut[(ii, probe_id)] for ii in range(data.shape[0])],
                    'time': np.arange(data.shape[1]) / self.probes.loc[probe_id]['sampling_rate']
                }
            )
            output_waveforms[uid] = output_waveforms[uid][output_waveforms[uid]["channel_id"] != -1]

        return output_waveforms


    def _build_inter_presentation_intervals(self):
        intervals = pd.DataFrame({
            'from_presentation_id': self.stimulus_presentations.index.values[:-1],
            'to_presentation_id': self.stimulus_presentations.index.values[1:],
            'interval': self.stimulus_presentations['start_time'].values[1:] - self.stimulus_presentations['stop_time'].values[:-1]
        })
        return intervals.set_index(['from_presentation_id', 'to_presentation_id'], inplace=False)


    def _filter_owned_df(self, key, ids=None, copy=True):
        df = getattr(self, key)

        if copy:
            df = df.copy()

        if ids is None:
            return df
        
        ids = warn_on_scalar(ids, f'a scalar ({ids}) was provided as ids, filtering to a single row of {key}.')

        df = df.loc[ids]

        if df.shape[0] == 0:
            warnings.warn(f'filtering to an empty set of {key}!')

        return df


    @classmethod
    def from_nwb_path(cls, path, nwb_version=2, api_kwargs=None, **kwargs):
        api_kwargs = {} if api_kwargs is None else api_kwargs
        # TODO: Is there a way for pynwb to check the file before actually loading it with io read? If so we could
        #       automatically check what NWB version is being inputed

        nwb_version = int(nwb_version)  # only use major version
        if nwb_version >= 2:
            NWBAdaptorCls = EcephysNwbSessionApi

        elif nwb_version == 1:
            NWBAdaptorCls = EcephysNwb1Api

        else:
            raise Exception(f'specified NWB version {nwb_version} not supported. Supported versions are: 2.X, 1.X')

        return cls(api=NWBAdaptorCls.from_path(path=path, **api_kwargs), **kwargs)


def build_spike_histogram(time_domain, spike_times, unit_ids, dtype=None, binarize=False):

    time_domain = np.array(time_domain)
    unit_ids = np.array(unit_ids)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, unit_ids.size), 
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]

    for ii, unit_id in enumerate(unit_ids):
        data = np.array(spike_times[unit_id])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts
    
    return tiled_data


def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1)) 
    domain += offsets[:, None]
    return callback(domain)


def removed_unused_stimulus_presentation_columns(stimulus_presentations):
    to_drop = []
    for cn in stimulus_presentations.columns:
        if np.all(stimulus_presentations[cn].isna()):
            to_drop.append(cn)
        elif np.all(stimulus_presentations[cn].astype(str).values == ''):
            to_drop.append(cn)
        elif np.all(stimulus_presentations[cn].astype(str).values == 'null'):
            to_drop.append(cn)
    return stimulus_presentations.drop(columns=to_drop)


def nan_intervals(array, nan_like=["null"]):
    """ find interval bounds (bounding consecutive identical values) in an array, which may contain nans

    Parameters
    -----------
    array : np.ndarray

    Returns
    -------
    np.ndarray : 
        start and end indices of detected intervals (one longer than the number of intervals)

    """

    intervals = [0]
    current = array[0]
    for ii, item in enumerate(array[1:]):
        if is_distinct_from(item, current):
            intervals.append(ii+1)
        current = item
    intervals.append(len(array))

    return np.unique(intervals)


def is_distinct_from(left, right):
    if type(left) != type(right):
        return True
    if pd.isna(left) and pd.isna(right):
        return False
    if left is None and right is None:
        return False

    return left != right


def array_intervals(array):
    """ find interval bounds (bounding consecutive identical values) in an array

    Parameters
    -----------
    array : np.ndarray

    Returns
    -------
    np.ndarray : 
        start and end indices of detected intervals (one longer than the number of intervals)

    """

    changes = np.flatnonzero(np.diff(array)) + 1
    return np.concatenate([ [0], changes, [len(array)] ])


def warn_on_scalar(value, message):
    if not isinstance(value, Collection) or isinstance(value, str):
        warnings.warn(message)
        return [value]
    return value
