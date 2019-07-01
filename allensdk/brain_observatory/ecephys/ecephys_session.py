import warnings
from collections.abc import Collection

import xarray as xr
import numpy as np
import pandas as pd

from allensdk.core.lazy_property import LazyPropertyMixin
from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi, EcephysNwbSessionApi, EcephysNwb1Api


NON_STIMULUS_PARAMETERS = tuple([
    'start_time',
    'stop_time',
    'duration',
    'stimulus_block'
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
                Unique integer identifier for this unit's peak channel (the channel on which this 
                unit's responses were greatest)
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


    def __init__(self, api, **kwargs):
        self.api: EcephysSessionApi = api

        self.ecephys_session_id = self.LazyProperty(self.api.get_ecephys_session_id)
        self.running_speed= self.LazyProperty(self.api.get_running_speed)
        self.mean_waveforms = self.LazyProperty(self.api.get_mean_waveforms, wrappers=[self._build_mean_waveforms])
        self.spike_times = self.LazyProperty(self.api.get_spike_times, wrappers=[self._build_spike_times])

        self.probes = self.LazyProperty(self.api.get_probes)
        self.channels = self.LazyProperty(self.api.get_channels)

        self.stimulus_presentations = self.LazyProperty(self.api.get_stimulus_presentations, wrappers=[self._build_stimulus_presentations])
        self.units = self.LazyProperty(self.api.get_units, wrappers=[self._build_units_table])
        self.inter_presentation_intervals = self.LazyProperty(self._build_inter_presentation_intervals)


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
        ''' Build a dataset of spike counts surrounding stimulus onset per unit and stimulus frame.

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
        xarray.Dataset :
            Contains a data array named spike_counts whose dimensions are stimulus presentation, unit, 
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
        tiled_data = np.zeros(
            (domain.shape[0], domain.shape[1], units.shape[0]), 
            dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
        )

        for ii, unit_id in enumerate(np.array(units.index.values)):
            data = self.spike_times[unit_id]
            flat_indices = np.searchsorted(domain.flat, data)

            unique, counts = np.unique(flat_indices, return_counts=True)
            valid = np.where( 
                (unique % len(bin_edges) != 0) 
                & (unique >= 0) 
                & (unique <= domain.size) 
            )

            unique = unique[valid]
            counts = counts[valid]

            tiled_data[:, :, ii].flat[unique] = counts > 0 if binarize else counts

        tiled_data = xr.DataArray(
            data=tiled_data[:, 1:, :], 
            coords={
                'stimulus_presentation_id': stimulus_presentations.index.values,
                'time_relative_to_stimulus_onset': bin_edges[:-1] + np.diff(bin_edges) / 2,
                'unit_id': units.index.values
            },
            dims=['stimulus_presentation_id', 'time_relative_to_stimulus_onset', 'unit_id']
        )

        return xr.Dataset(data_vars={'spike_counts': tiled_data})


    def presentationwise_spike_times(self, stimulus_presentation_ids=None, unit_ids=None):
        ''' Produce a table associating spike times with units and stimulus sweeos

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

        return pd.DataFrame({
            'stimulus_presentation_id': np.concatenate(presentation_ids).astype(int),
            'unit_id': np.concatenate(unit_ids).astype(int)
        }, index=pd.Index(np.concatenate(spike_times), name='spike_time')).sort_values('spike_time', axis=0)


    def conditionwise_spike_counts(self, stimulus_presentation_ids=None, unit_ids=None):
        ''' Count spikes by unit and stimulus condition

        Parameters
        ----------
        stimulus_presentation_ids : array-like
            Filter to these stimulus presentations
        unit_ids : array-like
            Filter to these units

        Returns
        -------
        pd.DataFrame :
            Each row describes a condition and unit. The 'count' column states the number 
            of spikes emitted by the unit under the condition.

        Notes
        -----
        A stimulus condition is a setting of the parameters for a particular stimulus.

        '''

        spike_times = self.presentationwise_spike_times(stimulus_presentation_ids=stimulus_presentation_ids, unit_ids=unit_ids)
        stimulus_presentations = self._filter_owned_df('stimulus_presentations', ids=stimulus_presentation_ids)
        return count_spikes_by_condition(spike_times, stimulus_presentations)


    def conditionwise_mean_spike_counts(self, stimulus_presentation_ids=None, unit_ids=None):
        ''' Report average spike counts by unit and stimulus condition

        Parameters
        ----------
        stimulus_presentation_ids : array-like
            Filter to these stimulus presentations
        unit_ids : array-like
            Filter to these units

        Returns
        -------
        pd.DataFrame :
            Each row describes a condition and unit. The 'mean_spike_count' column states the average
            number of spikes emitted by the unit under the condition.

        Notes
        -----
        A stimulus condition is a setting of the parameters for a particular stimulus.

        '''

        stimulus_presentations = self._filter_owned_df('stimulus_presentations', ids=stimulus_presentation_ids)
        spike_times = self.presentationwise_spike_times(stimulus_presentation_ids=stimulus_presentation_ids, unit_ids=unit_ids)
        return mean_spikes_by_condition(spike_times, stimulus_presentations)


    def get_stimulus_conditions(self, stimulus_presentation_ids=None):
        ''' Report stimulus conditions applied during this session

        Parameters
        ----------
        stimulus_presentation_ids : array-like
            Filter to these stimulus presentations

        Returns
        -------
        pd.DataFrame :
            Each row describes a condition

        Notes
        -----
        A stimulus condition is a setting of the parameters for a particular stimulus.

        '''

        stimulus_presentations = self._filter_owned_df('stimulus_presentations', ids=stimulus_presentation_ids)

        stimulus_presentations = stimulus_presentations.drop(columns=list(NON_STIMULUS_PARAMETERS))
        stimulus_conditions = stimulus_presentations.drop_duplicates()
        stimulus_conditions = stimulus_conditions.reset_index(inplace=False).drop(columns=['stimulus_presentation_id'])
        stimulus_conditions.index.name = 'stimulus_condition_index'

        stimulus_conditions = removed_unused_stimulus_presentation_columns(stimulus_conditions)
        return stimulus_conditions


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
            if drop_nulls:
                uniques = uniques[uniques != 'null']
            parameters[colname] = uniques

        return parameters


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

        # pandas groupby ops ignore nans, so we need a new null value that pandas does not recognize as null ...
        stimulus_presentations.loc[stimulus_presentations['stimulus_name'] == '', 'stimulus_name'] = 'spontaneous_activity'
        stimulus_presentations[stimulus_presentations == ''] = np.nan
        stimulus_presentations = stimulus_presentations.fillna('null') # 123 / 2**8

        stimulus_presentations['duration'] = stimulus_presentations['stop_time'] - stimulus_presentations['start_time']

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
            'manual_structure_id': 'structure_id',
            'manual_structure_acronym': 'structure_acronym',
            'local_index_channel': 'channel_local_index',
        })

        table = table.loc[
            (table['valid_data'])
            & (table['quality'] == 'good')
        ]

        # table = table.drop(columns=['local_index_unit', 'quality', 'valid_data'])
        return table.sort_values(by=['probe_description', 'probe_vertical_position', 'probe_horizontal_position'])


    def _build_nwb1_waveforms(self, mean_waveforms):
        # _build_mean_waveforms() assumes every unit has the same number of waveforms and that a unit-waveform exists
        # for all channels. This is not true for NWB 1 files where each unit has ONE waveform on ONE channel
        units_df = self.units
        output_waveforms = {}
        sampling_rate_lu = {uid: self.probes.loc[r['probe_id']]['sampling_rate'] for uid, r in units_df.iterrows()}

        for uid in list(mean_waveforms.keys()):
            # assert(sampling_rate_lu[uid] == 30000)
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
        # from ecephys_analysis_modules.modules.modality_comparison.ecephys_nwb1_adaptor import EcephysNwb1Adaptor
        if isinstance(self.api, EcephysNwb1Api):
            return self._build_nwb1_waveforms(mean_waveforms)

        # TODO: there is a bug either here or (more likely) in LIMS unit data ingest which causes the peak channel 
        # to be off by a few (exactly 1?) indices
        # we could easily recompute here, but better to fix it at the source
        channel_id_lut = {(row['local_index'], row['probe_id']): cid for cid, row in self.channels.iterrows()}
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

        return cls(api=NWBAdaptorCls.from_path(path=path, **api_kwargs), ** kwargs)


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


def count_by_condition(stimulus_presentations, exclude_parameters=NON_STIMULUS_PARAMETERS):
    
    stimulus_presentations =  stimulus_presentations.copy()
    stimulus_presentations = stimulus_presentations.drop(columns=list(exclude_parameters))
    
    cols = stimulus_presentations.columns.tolist()
    stimulus_presentations['count'] = 0
    return stimulus_presentations.groupby(cols, as_index=False).count()


def count_spikes_by_condition(spike_times, stimulus_presentations):
    spike_times = spike_times.copy()
    spike_times = spike_times.merge(stimulus_presentations, left_on='stimulus_presentation_id', right_index=True)
    spike_times = spike_times.drop(columns=['stimulus_presentation_id'])
    return count_by_condition(spike_times)


def mean_spikes_by_condition(spike_times, stimulus_presentations, non_stimulus_parameters=NON_STIMULUS_PARAMETERS):
    presentation_counts_by_condition = count_by_condition(stimulus_presentations)
    spike_counts_by_condition = count_spikes_by_condition(spike_times, stimulus_presentations)

    colnames = set(spike_counts_by_condition.columns.values) & set(presentation_counts_by_condition.columns.values)
    stimulus_parameters = [
        sp for sp in colnames
        if sp not in non_stimulus_parameters
        and sp != 'count'
    ]

    mean_spikes = spike_counts_by_condition.merge(
        presentation_counts_by_condition, 
        left_on=stimulus_parameters,
        right_on=stimulus_parameters,
        suffixes=['_spikes', '_presentations'],
        how='left'
    )
    
    mean_spikes['mean_spike_count'] = mean_spikes['count_spikes'] / mean_spikes['count_presentations']
    return mean_spikes.drop(columns=['count_spikes', 'count_presentations'])


def intervals_structures(table, structure_id_key="manual_structure_id", structure_label_key="manual_structure_acronym"):
    """ find on a channels / units table intervals of channels inserted into particular structures

    Parameters
    ----------
    table : pd.DataFrame
        A table of channels (or units, with peak channels)
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

    intervals = nan_intervals(table[structure_id_key])
    labels = table[structure_label_key][intervals[:-1]]

    return labels, intervals


def nan_intervals(array):
    """ find interval bounds (bounding consecutive identical values) in an array, which may contain nans

    Parameters
    -----------
    array : np.ndarray

    Returns
    -------
    np.ndarray : 
        start and end indices of detected intervals (one longer than the number of intervals)

    """
    return array_intervals(np.nan_to_num(array))


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
