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

        #print(self.ecephys_session.spike_times)
        #print(self.ecephys_session.units.columns)
        #print(self.ecephys_session.units[['location', 'structure_acronym']])
        #print(self.ecephys_session.units[])
        #exit()
        self._cell_ids = None
        self._cells_filter = {'location': 'probeC', 'structure_acronym': 'VISp'}
        self._number_cells = None
        self._spikes = None
        self._stim_table = None
        self._stim_table_spontaneous = None
        self._stimulus_names = None
        self._orivals = None
        self._number_ori = None
        self._sfvals = None
        self._number_sf = None
        self._phasevals = None
        self._number_phase = None
        self._running_speed = None
        self._sweep_events = None
        self._mean_sweep_events = None

    @property
    def ecephys_session(self):
        return self._ecephys_session

    @property
    def cell_id(self):
        # BOb analog
        if self._cell_ids is None:
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
        # BOb analog
        if not self._number_cells:
            self._number_cells = len(self.cell_id)
        return self._number_cells

    @property
    def spikes(self):
        if self._spikes:
            return self._spikes
        else:
            self._spikes = self.ecephys_session.spike_times
            if len(self._spikes) > self.numbercells:
                # if a filter has been applied st not all the cells are being used in the analysis
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

    @property
    def stim_table(self):
        # BOb analog
        if self._stim_table is None:
            # TODO: Give warning if no static_gratings stimulus
            # Older versions of NWB files the stimulus name is in the form stimulus_gratings_N, so if self._stimulus_names
            # is not explicity specified try to figure out stimulus
            if self._stimulus_names is None:
                stims_table = self.ecephys_session.stimulus_presentations
                stim_names = [s for s in stims_table['stimulus_name'].unique()
                              if s.lower().startswith('static_gratings')]

                self._stim_table = stims_table[stims_table['stimulus_name'].isin(stim_names)]

            else:
                self._stimulus_names = [self._stimulus_names] if isinstance(self._stimulus_names, string_types) else self._stimulus_names
                self._stim_table = self.ecephys_session.get_presentations_for_stimulus(self._stimulus_names)

        return self._stim_table

    @property
    def stim_table_spontaneous(self):
        # BOb analog
        if self._stim_table_spontaneous is None:
            # TODO: The original version filtered out stims of len < 100, figure out why or if this value should be user-defined?
            stim_table = self.ecephys_session.get_presentations_for_stimulus(['spontaneous'])
            self._stim_table_spontaneous = stim_table[stim_table['duration'] > 100.0]

        return self._stim_table_spontaneous

    @property
    def orivals(self):
        if self._orivals is None:
            self._get_stim_table_stats()

        return self._orivals

    @property
    def number_ori(self):
        if self._number_ori is None:
            self._get_stim_table_stats()

        return self._number_ori

    @property
    def sfvals(self):
        if self._sfvals is None:
            self._get_stim_table_stats()

        return self._sfvals

    @property
    def number_sf(self):
        if self._number_sf is None:
            self._get_stim_table_stats()

        return self._number_sf

    @property
    def phasevals(self):
        if self._phasevals is None:
            self._get_stim_table_stats()

        return self._phasevals

    @property
    def number_phase(self):
        if self._number_phase is None:
            self._get_stim_table_stats()

        return self._number_phase

    @property
    def sweep_events(self):
        if self._sweep_events is None:
            # stim_presentation_ids = self.stim_table.index.values
            # unit_ids = self.cell_id
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
            # running_speed = pd.DataFrame(index=self.stim_table.index.values, columns=['running_speed'])
            stim_times = np.zeros(len(self.stim_table)*2, dtype=np.float64)
            stim_times[::2] = self.stim_table['start_time'].values
            stim_times[1::2] = self.stim_table['stop_time'].values
            sampled_indicies = np.where((self.dxtime >= stim_times[0])&(self.dxtime <= stim_times[-1]))[0]
            relevant_dxtimes = self.dxtime[sampled_indicies] # self.dxtime[(self.dxtime >= stim_times[0])&(self.dxtime <= stim_times[-1])]
            relevant_dxcms = self.dxcm[sampled_indicies]

            indices = np.searchsorted(stim_times, relevant_dxtimes) - 1  # excludes dxtimes occuring at time_stop
            rs_tmp_df = pd.DataFrame({'running_speed': relevant_dxcms, 'stim_indicies': indices})
            rs_tmp_df = rs_tmp_df.groupby('stim_indicies').agg('mean')

            # Remove odd numbered indicies (which indicates that a running speed was measured between start and stop times).
            rs_tmp_df = rs_tmp_df.loc[list(range(0, 12000, 2))]
            rs_tmp_df = rs_tmp_df.set_index(self.stim_table.index.values)
            self._running_speed = rs_tmp_df

        return self._running_speed

    @property
    def mean_sweep_events(self):
        raise NotImplementedError()


    def _get_stim_table_stats(self):
        sg_stim_table = self.stim_table
        self._orivals = np.sort(sg_stim_table['Ori'].dropna().unique())  # list(range(0, 180, 30))
        self._number_ori = len(self._orivals)

        self._sfvals = np.sort(sg_stim_table['SF'].dropna().unique())  # [0.02, 0.04, 0.08, 0.16, 0.32]
        self._number_sf = len(self._sfvals)

        self._phasevals = np.sort(sg_stim_table['Phase'].dropna().unique())  # [0.0, 0.25, 0.50, 0.75]
        self._number_phase = len(self._phasevals)
