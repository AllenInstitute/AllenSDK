from typing import NamedTuple

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
    running_speed
    mean_waveforms
    spike_times
    stimulus_table
    units_table

    '''

    def __init__(self, api, **kwargs):
        self.api: EcephysApi  = api

        self.running_speed= self.LazyProperty(self.api.get_running_speed)
        self.mean_waveforms = self.LazyProperty(self.api.get_mean_waveforms)
        self.spike_times = self.LazyProperty(self.api.get_spike_times)

        self.probes = self.LazyProperty(self.api.get_probes)
        self.channels = self.LazyProperty(self.api.get_channels)

        self.stimulus_table = self.LazyProperty(self.api.get_stimulus_table, wrappers=[self._build_stimulus_table])
        self.units_table = self.LazyProperty(self.api.get_units, wrappers=[self._build_units_table])


    def filter_stimulus_frames(self, cond, table=None, drop_irrelevant_data=True, *args, **kwargs):
        table = self.stimulus_table if table is None else table
        kwargs['drop'] = kwargs.get('drop', True)
        table = table.where(cond, *args, **kwargs)
        table = clean_stimulus_dataset_arrays(table)
        return table

    
    def filter_units_table(self, cond, table=None, *args, **kwargs):
        table = self.units_table if table is None else table
        kwargs['drop'] = kwargs.get('drop', True)
        return table.where(cond, *args, **kwargs)


    def framewise_spike_counts(self, bin_edges,  stimulus_frames, units, binarize=False, dtype=None):
        ''' Build a dataset of spike counts surrounding stimulus onset per unit and stimulus frame.
        '''

        bin_edges = np.array(bin_edges)
        domain = build_time_window_domain(bin_edges, stimulus_frames['start_time'].values)
        tiled_data = np.zeros(
            (domain.shape[0], domain.shape[1], len(units['unit_id'])), 
            dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
        )

        for ii, unit_id in enumerate(np.array(units['unit_id'])):
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
                'stimulus_frame_id': stimulus_frames['stimulus_frame_id'],
                'time_relative_to_stimulus_onset': bin_edges[:-1] + np.diff(bin_edges) / 2,
                'unit_id': units['unit_id']
            },
            dims=['stimulus_frame_id', 'time_relative_to_stimulus_onset', 'unit_id']
        )

        return xr.Dataset(data_vars={'spike_counts': tiled_data})


    def framewise_spike_times(self, stimulus_frames, units):
        '''
        '''

        frame_times = np.zeros([len(stimulus_frames['stimulus_frame_id']) * 2])
        frame_times[::2] = np.array(stimulus_frames['start_time'])
        frame_times[1::2] = np.array(stimulus_frames['stop_time'])
        all_frame_ids = np.array(stimulus_frames['stimulus_frame_id'])

        frame_ids = []
        unit_ids = []
        spike_times = []

        for ii, unit_id in enumerate(np.array(units['unit_id'])):
            data = self.spike_times[unit_id]
            indices = np.searchsorted(frame_times, data) - 1

            index_valid = indices % 2 == 0
            frames = all_frame_ids[np.floor(indices / 2).astype(int)]

            sorder = np.argsort(frames)
            frames = frames[sorder]
            index_valid = index_valid[sorder]
            data = data[sorder]

            changes = np.where(np.ediff1d(frames, to_begin=1, to_end=1))[0]
            for ii, jj in zip(changes[:-1], changes[1:]):
                values = data[ii:jj][index_valid[ii:jj]]
                if values.size == 0:
                    continue

                unit_ids.append(np.zeros([values.size]) + unit_id)
                frame_ids.append(np.zeros([values.size]) + frames[ii])
                spike_times.append(values)

        return pd.DataFrame({
            'stimulus_frame_id': np.concatenate(frame_ids).astype(int),
            'unit_id': np.concatenate(unit_ids).astype(int)
        }, index=pd.Index(np.concatenate(spike_times), name='spike_time'))


    def enumerate_stimulus_conditions(self, query=None):
        stimulus_table = optionally_query_dataframe(self.stimulus_table, query)
        stimulus_table = stimulus_table.drop(columns=['start_time', 'stop_time', 'stimulus_block'])
        stimulus_table = stimulus_table.drop_duplicates()
        stimulus_table = stimulus_table.reset_index(inplace=False).drop(columns=['id'])

        stimulus_table = clean_stimulus_table_columns(stimulus_table)
        return stimulus_table


    def distinct_stimulus_parameter_values(self, query=None):
        stimulus_table = optionally_query_dataframe(self.stimulus_table, query)
        stimulus_table = stimulus_table.drop(columns=['start_time', 'stop_time', 'stimulus_block'])
        return {col: stimulus_table[col].unique() for col in stimulus_table.columns}


    def _build_stimulus_table(self, stimulus_table):
        stimulus_table = stimulus_table.copy()
        stimulus_table.index.name  = 'stimulus_frame_id'
        return stimulus_table.to_xarray()


    def _build_units_table(self, units_table):
        channels = self.channels.copy()
        probes = self.probes.copy()

        table = pd.merge(units_table, channels, left_on='peak_channel_id', right_index=True, suffixes=['_unit', '_channel'])
        table = pd.merge(table, probes, left_on='probe_id', right_index=True, suffixes=['_unit', '_probe'])
        table.index.name = 'unit_id'
        return table.sort_values(by=['description', 'probe_vertical_position', 'probe_vertical_position']).to_xarray()


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


class StimulusFrameUnitIndex(NamedTuple):
    stimulus_frame_id: int
    unit_id: int


def clean_stimulus_table_columns(stimulus_table):
    to_drop = []
    for cn in stimulus_table.columns:
        if np.all(stimulus_table[cn].isna()):
            to_drop.append(cn)
        elif np.all(stimulus_table[cn].astype(str).values == ''):
            to_drop.append(cn)
    return stimulus_table.drop(columns=to_drop)


def clean_stimulus_dataset_arrays(dataset):
    to_drop = []
    for da in dataset.data_vars:
        if np.issubdtype(dataset[da].dtype, np.floating) and np.all(np.isnan(dataset[da])):
            to_drop.append(da)
        elif np.all(dataset[da] == ''):
            to_drop.append(da)
    return dataset.drop(to_drop)