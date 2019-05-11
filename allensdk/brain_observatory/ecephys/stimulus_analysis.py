from six import string_types
import numpy as np
import pandas as pd

from .ecephys_session import EcephysSession


class StimulusAnalysis(object):
    def __init__(self, ecephys_session, **kwargs):
        if isinstance(ecephys_session, EcephysSession):
            self._ecephys_session = ecephys_session
        elif isinstance(ecephys_session, string_types):
            self._ecephys_session = EcephysSession.from_nwb1_path(ecephys_session)

        self._cell_ids = None
        self._cells_filter = {'location': 'probeC', 'structure_acronym': 'VISp'}
        self._number_cells = None
        self._spikes = None
        self._stim_table = None
        self._stim_table_spontaneous = None
        self._stimulus_names = None
        self._running_speed = None
        self._sweep_events = None
        self._mean_sweep_events = None

    @property
    def ecephys_session(self):
        return self._ecephys_session

    @property
    def cell_id(self):
        """Returns a list of unit-ids for which to apply the analysis"""
        # BOb analog
        if self._cell_ids is None:
            # Original analysis files was hardcoded that only cells from probeC/VISp, replaced with a filter dict.
            # TODO: Remove filter if it's not nessecary, or else make cells_filter a class parameter.
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
        # BOb analog
        return self.ecephys_session.running_speed.values

    @property
    def dxtime(self):
        # BOb analog
        return self._ecephys_session.running_speed.timestamps

    def _stim_presentation_prefix(self):
        raise NotImplementedError()

    @property
    def stim_table(self):
        # BOb analog
        # Stimulus table is already in EcephysSession object, just need to subselect 'static_gratings' presentations.
        if self._stim_table is None:
            # TODO: Give warning if no static_gratings stimulus?
            if self._stimulus_names is None:
                # Older versions of NWB files the stimulus name is in the form stimulus_gratings_N, so if
                # self._stimulus_names is not explicity specified try to figure out stimulus
                stims_table = self.ecephys_session.stimulus_presentations
                stim_names = [s for s in stims_table['stimulus_name'].unique()
                              if s.lower().startswith('static_gratings')]

                self._stim_table = stims_table[stims_table['stimulus_name'].isin(stim_names)]

            else:
                self._stimulus_names = [self._stimulus_names] if isinstance(self._stimulus_names, string_types) \
                    else self._stimulus_names
                self._stim_table = self.ecephys_session.get_presentations_for_stimulus(self._stimulus_names)

        return self._stim_table

    @property
    def stim_table_spontaneous(self):
        # BOb analog
        # Returns the stimulus table with only 'spanteous' presentations. Used by sweep_p_events for creating null dist.
        if self._stim_table_spontaneous is None:
            # TODO: The original version filtered out stims of len < 100, figure out why or if this value should
            #   be user-defined?
            stim_table = self.ecephys_session.get_presentations_for_stimulus(['spontaneous'])
            self._stim_table_spontaneous = stim_table[stim_table['duration'] > 100.0]

        return self._stim_table_spontaneous

    @property
    def sweep_events(self):
        if self._sweep_events is None:
            start_times = self.stim_table['start_time'].values - 1.0
            stop_times = self.stim_table['stop_time'].values
            sweep_events = pd.DataFrame(index=self.stim_table.index.values, columns=self.spikes.keys())

            for specimen_id, spikes in self.spikes.items():
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
            relevant_dxtimes = self.dxtime[sampled_indicies] # self.dxtime[(self.dxtime >= stim_times[0])&(self.dxtime <= stim_times[-1])]
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
        raise NotImplementedError()
