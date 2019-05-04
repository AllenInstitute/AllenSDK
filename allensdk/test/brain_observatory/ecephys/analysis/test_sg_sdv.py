from ecephys_analysis_modules.common.static_gratings_sdv import StaticGratings
import os
import h5py
import numpy as np
import pandas as pd
import itertools


# nwb_path = '/allen/aibs/mat/Kael/ecephys_data/mouse387858.spikes.nwb'
mouseid = 'mouse412792'
stimulus = 'static_grating'
nwb_path = '/allen/aibs/mat/Kael/ecephys_data/{}.spikes.nwb'.format(mouseid)


def save_results(mouseid='mouse412792'):
    sg = StaticGratings(nwb_path=nwb_path)
    with h5py.File('expected/{}.{}.h5'.format(mouseid, stimulus), 'w') as h5:
        h5.attrs['numbercells'] = sg.numbercells

        spikes_grp = h5.create_group('/spikes')
        for specimen_id, spikes in sg.spikes.items():
            cell_grp = spikes_grp.create_group(str(specimen_id))
            cell_grp.create_dataset('spikes', data=spikes)

        stim_table_grp = h5.create_group('/stimulus_table')
        for col in sg.stim_table.columns:
            stim_table_grp.create_dataset(col, data=sg.stim_table[col])

        h5.create_dataset('dxcm', data=sg.dxcm)
        h5.create_dataset('dxcm_ts', data=sg.dxcm_ts)

        spon_stim_table_grp = h5.create_group('/stimulus_table_spontaneous')
        for col in sg.stim_table_sp.columns:
            spon_stim_table_grp.create_dataset(col, data=sg.stim_table_sp[col])

        save_sweep_data(h5, sg.sweep_events)

        mse_grp = h5.create_group('mean_sweep_events')
        mse_grp.create_dataset('specimen_ids', data=[int(i) for i in sg.mean_sweep_events.columns.values])
        mse_grp.create_dataset('presentation_ids', data=[int(i) for i in sg.mean_sweep_events.index.values])
        mse_grp.create_dataset('data', data=sg.mean_sweep_events.values)

        sweep_p_vals = h5.create_group('sweep_p_values')
        sweep_p_vals.create_dataset('specimen_ids', data=[int(i) for i in sg.sweep_p_values.columns.values])
        sweep_p_vals.create_dataset('presentation_ids', data=[int(i) for i in sg.sweep_p_values.index.values])
        sweep_p_vals.create_dataset('data', data=sg.sweep_p_values.values)

        # h5.create_dataset('running_speed', data=sg.running_speed['running_speed'])

        # TODO: Adding response_events and response_trials

        save_peak_data(h5, sg.peak)


def save_sweep_data(h5, sweeps_df):
    presentation_ids = sweeps_df.index.values
    specimen_ids = [int(sid) for sid in sweeps_df.columns.values]
    sid_lu = {sid: i for i, sid in enumerate(specimen_ids)}

    se_grp = h5.create_group('sweep_events')
    se_grp.create_dataset('specimen_ids', data=specimen_ids, dtype=np.uint64)
    events_table = se_grp.create_dataset('events_table', (len(presentation_ids), len(specimen_ids)),
                                         dtype=h5py.special_dtype(ref=h5py.RegionReference))

    for sid in specimen_ids:
        spikes_array = np.array([], dtype=np.float)
        indicies = [0]
        for pid in presentation_ids:
            c_array = sweeps_df.iloc[pid][str(sid)]
            # c_array = np.fromstring(sweeps_df.iloc[pid][str(sid)][1:-1], sep=' ')
            spikes_array = np.append(spikes_array, c_array)
            indicies.append(indicies[-1] + len(c_array))

        s_spikes = se_grp.create_dataset('{}/data'.format(sid), data=spikes_array)

        col_indx = sid_lu[sid]
        for i in range(len(indicies) - 1):
            i_beg = indicies[i]
            i_end = indicies[i + 1]
            regref = s_spikes.regionref[i_beg:i_end]
            events_table[i, col_indx] = regref


def save_peak_data(h5, peak_df):
    dtype_lu = {
        'cell_specimen_id': np.uint64,
        'responsive_sg': bool,
        'num_pref_trials_sg': np.uint64,
        # 'fit_sf_ind_sg': np.uint64
    }

    peak_grp = h5.create_group('peak')
    for col in peak_df.columns:
        print(col)
        dtype = dtype_lu.get(col, np.float64)
        peak_grp.create_dataset(col, data=peak_df[col].astype(dtype))


def test_results(mouseid):
    sg = StaticGratings(nwb_path=nwb_path)


def save_events():
    sweep_events_df = pd.read_csv('sweep_events.csv', sep=' ', index_col=0)
    presentation_ids = sweep_events_df.index.values
    specimen_ids = [int(sid) for sid in sweep_events_df.columns.values]
    #presentation_ids = range(0, 200)
    #specimen_ids = [216, 217, 215, 212, 213, 210, 211, 218, 219, 164, 165, 166, 167, 161, 162, 163, 262, 260, 266, 264, 199, 198, 195, 194, 197, 196, 191]
    # print(specimen_ids)

    sid_lu = {sid: i for i, sid in enumerate(specimen_ids)}

    with h5py.File('expected/{}.{}.h5'.format(mouseid, stimulus), 'w') as h5:
        se_grp = h5.create_group('sweep_events')
        se_grp.create_dataset('specimen_ids', data=specimen_ids, dtype=np.uint64)
        # events_grp = se_grp.create_group('spike_events')
        events_table = se_grp.create_dataset('events_table', (len(presentation_ids), len(specimen_ids)),
                                             dtype=h5py.special_dtype(ref=h5py.RegionReference))

        for sid in specimen_ids:
            spikes_array = np.array([], dtype=np.float)
            indicies = [0]
            for pid in presentation_ids:
                c_array = np.fromstring(sweep_events_df.iloc[pid][str(sid)][1:-1], sep=' ')
                spikes_array = np.append(spikes_array, c_array)
                indicies.append(indicies[-1] + len(c_array))

            #print len(spikes_array)
            #print indicies
            #exit()
            s_spikes = se_grp.create_dataset('{}/data'.format(sid), data=spikes_array)

            col_indx = sid_lu[sid]
            for i in range(len(indicies)-1):
                i_beg = indicies[i]
                i_end = indicies[i+1]
                regref = s_spikes.regionref[i_beg:i_end]
                # print(regref)
                events_table[i, col_indx] = regref

    with h5py.File('expected/{}.{}.h5'.format(mouseid, stimulus), 'r') as h5:
        regionref = h5['/sweep_events/events_table'][0,0]
        print(regionref)
        print(h5['/sweep_events/216/data'][regionref])



if __name__ == '__main__':
    # save_events()
    save_results()
    #peak_df = pd.read_csv('peak.csv', sep=' ', index_col=0)
    #print(peak_df.iloc[23])
