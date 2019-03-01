import warnings

import xarray as xr
import numpy as np
import pandas as pd

from allensdk.core.lazy_property import LazyPropertyMixin
from allensdk.brain_observatory.ecephys.ecephys_api import EcephysApi, EcephysNwbApi
from . import RunningSpeed

class EcephysSession(LazyPropertyMixin):
    ''' Represents data from a single EcephysSession

    Attributes
    ----------
    units : pd.Dataframe
        A table whose rows are sorted units (putative neurons) and whose columns are characteristics of those units.
        Index is:
            unit_id : int
                Unique integer identifier for this unit.
        Columns are:
            firing_rate : float
                This unit's firing rate (spikes / s) calculated over the window of that unit's activity (the time from 
                its first detected spike to its last).
            isi_violations : float
                Estamate of this unit's contamination rate (larger means that more of the spikes assigned to this unit 
                probably originated from other neurons). Calculated as a ratio of the firing rate of the unit over 
                periods where spikes would be isi-violating vs the total firing rate of the unit.
            peak_channel_id : int
                Unique integer identifier for this unit's peak channel (the channel on which this unit's responses 
                were greatest)
            snr : float
                Signal to noise ratio for this unit.
            probe_horizontal_position :  numeric
                The horizontal (short-axis) position of this unit's peak channel in microns.
            probe_vertical_position : numeric
                The vertical (long-axis, lower values are closer to the probe base) position of this unit's peak 
                channel in microns.
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
    stimulus_sweeps : pd.DataFrame
        Table whose rows are stimulus sweeps and whose columns are sweep characteristics. A stimulus sweep is the 
        smallest unit of distinct stimulus presentation and lasts for (usually) 1 60hz frame.
        Index is
            stimulus_sweep_id : int
                Unique identifier for this stimulus sweep
        Columns are
            start_time :  float
                Time (s) at which this sweep began
            stop_time : float
                Time (s) at which this sweep ended
            stimulus_name : str
                Identifies the stimulus family (e.g. "drifting_gratings" or "natural_movie_3") used for this sweep. The 
                stimulus family, along with relevant parameter values, provides the information required to reconstruct 
                the stimulus presented during this sweep. The empty string indicates a blank period.
            stimulus_block : numeric
                A stimulus block is made by sequentially presenting sweeps from the same stimulus family. This value is 
                the index of the block which contains this sweep. During a blank period, this is NaN.
            TF : float
                Temporal frequency, or NaN when not appropriate.
            SF : float
                Spatial frequency, or NaN when not appropriate
            Ori : float
                Orientation (in degress) or NaN when not appropriate
            Contrast : float
            Pos_x : float
            Pos_y : float
            Color : numeric
            Image : numeric
            Phase : float

    '''

    def __init__(self, api, **kwargs):
        self.api: EcephysApi  = api

        self.running_speed= self.LazyProperty(self.api.get_running_speed)
        self.mean_waveforms = self.LazyProperty(self.api.get_mean_waveforms, wrappers=[self._build_mean_waveforms])
        self.spike_times = self.LazyProperty(self.api.get_spike_times)

        self.probes = self.LazyProperty(self.api.get_probes)
        self.channels = self.LazyProperty(self.api.get_channels)

        self.stimulus_sweeps = self.LazyProperty(self.api.get_stimulus_table, wrappers=[self._build_stimulus_sweeps])
        self.units = self.LazyProperty(self.api.get_units, wrappers=[self._build_units_table])


    def framewise_spike_counts(self, bin_edges, stimulus_sweeps, units, binarize=False, dtype=None, large_bin_size_threshold=0.001):
        ''' Build a dataset of spike counts surrounding stimulus onset per unit and stimulus frame.
        '''

        largest_bin_size = np.amax(np.diff(bin_edges))
        if binarize and largest_bin_size  > large_bin_size_threshold:
            warnings.warn(
                f'You\'ve elected to binarize spike counts, but your maximum bin width is {largest_bin_size:2.5f} seconds. '
                'Binarizing spike counts with such a large bin width can cause significant loss of accuracy! '
                f'Please consider only binarizing spike counts when your bins are <= {large_bin_size_threshold} seconds wide.'
            )

        stimulus_sweeps = stimulus_sweeps.copy()
        units = units.copy()

        bin_edges = np.array(bin_edges)
        domain = build_time_window_domain(bin_edges, stimulus_sweeps['start_time'].values)
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
                'stimulus_sweep_id': stimulus_sweeps.index.values,
                'time_relative_to_stimulus_onset': bin_edges[:-1] + np.diff(bin_edges) / 2,
                'unit_id': units.index.values
            },
            dims=['stimulus_sweep_id', 'time_relative_to_stimulus_onset', 'unit_id']
        )

        return xr.Dataset(data_vars={'spike_counts': tiled_data})


    def sweepwise_spike_times(self, stimulus_sweeps, units):
        '''
        '''

        sweep_times = np.zeros([stimulus_sweeps.shape[0] * 2])
        sweep_times[::2] = np.array(stimulus_sweeps['start_time'])
        sweep_times[1::2] = np.array(stimulus_sweeps['stop_time'])
        all_sweep_ids = np.array(stimulus_sweeps.index.values)

        sweep_ids = []
        unit_ids = []
        spike_times = []

        for ii, unit_id in enumerate(units.index.values):
            data = self.spike_times[unit_id]
            indices = np.searchsorted(sweep_times, data) - 1

            index_valid = indices % 2 == 0
            sweeps = all_sweep_ids[np.floor(indices / 2).astype(int)]

            sorder = np.argsort(sweeps)
            sweeps = sweeps[sorder]
            index_valid = index_valid[sorder]
            data = data[sorder]

            changes = np.where(np.ediff1d(sweeps, to_begin=1, to_end=1))[0]
            for ii, jj in zip(changes[:-1], changes[1:]):
                values = data[ii:jj][index_valid[ii:jj]]
                if values.size == 0:
                    continue

                unit_ids.append(np.zeros([values.size]) + unit_id)
                sweep_ids.append(np.zeros([values.size]) + sweeps[ii])
                spike_times.append(values)

        return pd.DataFrame({
            'stimulus_sweep_id': np.concatenate(sweep_ids).astype(int),
            'unit_id': np.concatenate(unit_ids).astype(int)
        }, index=pd.Index(np.concatenate(spike_times), name='spike_time'))


    def spike_counts_by_unit_and_stimulus_sweep(self, stimulus_sweeps, units):
        spike_times = self.sweepwise_spike_times(stimulus_sweeps=stimulus_sweeps, units=units)
        groupby_cols = list(spike_times.columns)
        spike_times['count'] = 0.0
        return pd.DataFrame(spike_times.groupby(groupby_cols, as_index=False).count())


    def mean_spike_counts_by_unit_and_stimulus_condition(self, stimulus_sweeps, units):
        mean_counts = self.spike_counts_by_unit_and_stimulus_sweep(stimulus_sweeps=stimulus_sweeps, units=units)
        mean_counts.rename(columns={'count': 'mean_spike_count'}, inplace=True)

        mean_counts = mean_counts.reset_index().set_index(keys='stimulus_sweep_id')
        stimulus_sweeps = stimulus_sweeps.reindex(index=mean_counts.index)
        mean_counts = pd.concat([mean_counts, stimulus_sweeps], axis=1)
        mean_counts = mean_counts.drop(columns=['start_time', 'stop_time', 'index'])

        groupby_cols = [colname for colname in list(mean_counts.columns) if colname != 'mean_spike_count']
        mean_counts = mean_counts.groupby(groupby_cols, as_index=False).mean()
        mean_counts = mean_counts.set_index(keys='unit_id')
        return mean_counts

    def list_stimulus_conditions(self, stimulus_sweeps=None):
        if stimulus_sweeps is None:
            stimulus_sweeps = self.stimulus_sweeps.copy()

        stimulus_sweeps = stimulus_sweeps.drop(columns=['start_time', 'stop_time', 'stimulus_block'])
        stimulus_sweeps = stimulus_sweeps.drop_duplicates()
        stimulus_sweeps = stimulus_sweeps.reset_index(inplace=False).drop(columns=['stimulus_sweep_id'])

        stimulus_sweeps = removed_unused_stimulus_sweep_columns(stimulus_sweeps)
        return stimulus_sweeps


    def list_stimulus_parameter_values(self, stimulus_sweeps=None):
        if stimulus_sweeps is None:
            stimulus_sweeps = self.stimulus_sweeps.copy()

        stimulus_sweeps = stimulus_sweeps.drop(columns=['start_time', 'stop_time', 'stimulus_block'])
        return {col: stimulus_sweeps[col].unique() for col in stimulus_sweeps.columns}


    def _build_stimulus_sweeps(self, stimulus_sweeps):
        stimulus_sweeps.index.name = 'stimulus_sweep_id'
        return stimulus_sweeps

    def _build_units_table(self, units_table):
        channels = self.channels.copy()
        probes = self.probes.copy()

        self._unmerged_units = units_table.copy()
        table = pd.merge(units_table, channels, left_on='peak_channel_id', right_index=True, suffixes=['_unit', '_channel'])
        table = pd.merge(table, probes, left_on='probe_id', right_index=True, suffixes=['_unit', '_probe'])

        table.index.name = 'unit_id'
        table = table.rename(columns={'description': 'probe_description'})

        table = table.loc[
            (table['valid_data'])
            & (table['quality'] == 'good')
        ]
        table = table.drop(columns=['local_index_unit', 'local_index_channel', 'quality', 'valid_data'])
        return table.sort_values(by=['probe_description', 'probe_vertical_position', 'probe_vertical_position'])


    def _build_mean_waveforms(self, mean_waveforms):
        channels = self.channels.copy().sort_values(by='local_index').index.values
        for unit_id in list(mean_waveforms.keys()):
            mean_waveforms[unit_id] = xr.DataArray(
                data=mean_waveforms[unit_id],
                dims=['samples', 'channels'], # TODO: this ought to be in a time base
                coords={
                    'channels': channels,
                    'samples': np.arange(mean_waveforms[unit_id].shape[0])
                }
            )


    @classmethod
    def from_nwb_path(cls, path, api_kwargs=None, **kwargs):
        api_kwargs = {} if api_kwargs is None else api_kwargs
        return cls(api=EcephysNwbApi(path=path, **api_kwargs), **kwargs)


def build_time_window_domain(bin_edges, offsets):
    domain = np.tile(bin_edges[None, :], (len(offsets), 1)) 
    domain += offsets[:, None]
    return domain


def optionally_query_dataframe(df, query=None, inplace=False):
    if not inplace:    
        df = df.copy()
    if query is not None:
        df = df.query(query, inplace=inplace)
    return df


def df_to_xarray_named_index(df, name, copy=True):
    df = df.copy() if copy else df
    df.index.name = name
    return df.to_xarray()


def removed_unused_stimulus_sweep_columns(stimulus_sweeps):
    to_drop = []
    for cn in stimulus_sweeps.columns:
        if np.all(stimulus_sweeps[cn].isna()):
            to_drop.append(cn)
        elif np.all(stimulus_sweeps[cn].astype(str).values == ''):
            to_drop.append(cn)
    return stimulus_sweeps.drop(columns=to_drop)


def clean_stimulus_dataset_arrays(dataset):
    to_drop = []
    for da in dataset.data_vars:
        if np.issubdtype(dataset[da].dtype, np.floating) and np.all(np.isnan(dataset[da])):
            to_drop.append(da)
        elif np.all(dataset[da] == ''):
            to_drop.append(da)
    return dataset.drop(to_drop)
