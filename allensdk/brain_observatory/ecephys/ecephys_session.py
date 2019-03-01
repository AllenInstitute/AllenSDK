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
        A table whose rows are sorted units (putative neurons) and whose columns are characteristics of those units. The 
        values of this table's index are unique integer identifiers for each unit.
        Columns are:
            firing_rate	
            isi_violations	
            local_index_unit	
            peak_channel_id	
            quality	
            snr	
            local_index_channel	
            probe_horizontal_position	
            probe_id	
            probe_vertical_position	
            valid_data	
            description	
            location
    spike_times : dict
        A di
    running_speed
    mean_waveforms
    
    stimulus_sweeps
    units

    '''

    def __init__(self, api, **kwargs):
        self.api: EcephysApi  = api

        self.running_speed= self.LazyProperty(self.api.get_running_speed)
        self.mean_waveforms = self.LazyProperty(self.api.get_mean_waveforms)
        self.spike_times = self.LazyProperty(self.api.get_spike_times)

        self.probes = self.LazyProperty(self.api.get_probes)
        self.channels = self.LazyProperty(self.api.get_channels)

        self.stimulus_sweeps = self.LazyProperty(self.api.get_stimulus_table)
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



    # def stimulus_conditionwise_spike_summary(self, stimulus_sweeps, units, fn=np.mean):
    #     mean_counts = _spike_counts_by_stimulus_sweep_and_condition

    #     mean_counts.rename(columns={'count': 'mean_spike_count'}, inplace=True)
    #     mean_counts = mean_counts.groupby(['stimulus_index', 'stimulus_name', 'TF', 'SF', 'Ori', 'Contrast', 'unit_id'], as_index=False).mean()
    #     mean_counts = mean_counts.drop(columns='stimulus_sweep_id')
    #     mean_counts = mean_counts.set_index(keys='unit_id')



    # def enumerate_stimulus_conditions(self, query=None):
    #     stimulus_table = optionally_query_dataframe(self.stimulus_table, query)
    #     stimulus_table = stimulus_table.drop(columns=['start_time', 'stop_time', 'stimulus_block'])
    #     stimulus_table = stimulus_table.drop_duplicates()
    #     stimulus_table = stimulus_table.reset_index(inplace=False).drop(columns=['id'])

    #     stimulus_table = clean_stimulus_table_columns(stimulus_table)
    #     return stimulus_table


    # def distinct_stimulus_parameter_values(self, query=None):
    #     stimulus_table = optionally_query_dataframe(self.stimulus_table, query)
    #     stimulus_table = stimulus_table.drop(columns=['start_time', 'stop_time', 'stimulus_block'])
    #     return {col: stimulus_table[col].unique() for col in stimulus_table.columns}


    def _build_units_table(self, units_table):
        channels = self.channels.copy()
        probes = self.probes.copy()

        self._unmerged_units = units_table.copy()
        table = pd.merge(units_table, channels, left_on='peak_channel_id', right_index=True, suffixes=['_unit', '_channel'])
        table = pd.merge(table, probes, left_on='probe_id', right_index=True, suffixes=['_unit', '_probe'])
        table.index.name = 'unit_id'
        return table.sort_values(by=['description', 'probe_vertical_position', 'probe_vertical_position'])


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


# def merge_spike_times_and_stimulus_sweeps(
#     spike_times, stimulus_sweeps, 
#     stimulus_sweep_id_colname='stimulus_sweep_id', spike_time_colname='spike_time',
#     drop_from_merged=('start_time', 'stop_time', 'stimulus_block')
# ):
#     drop_from_merged = [drop_from_merged] if isinstance(drop_from_merged, str) else drop_from_merged

#     spike_times = spike_times.reset_index().set_index(keys=stimulus_sweep_id_colname)
#     stimulus_sweeps.index.name = stimulus_sweep_id_colname
#     stimulus_sweeps = stimulus_sweeps.reindex(index=spike_times.index)

#     merged = pd.concat([spike_times, stimulus_sweeps], axis=1)
#     merged = merged.drop(columns=list(drop_from_merged))
#     return merged.reset_index().set_index(keys=spike_time_colname)