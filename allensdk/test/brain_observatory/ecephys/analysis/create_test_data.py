from ecephys_analysis_modules.common.static_gratings_sdv import StaticGratings
from ecephys_analysis_modules.common.drifting_gratings_sdv import DriftingGratings
from ecephys_analysis_modules.common.natural_scenes_sdv import NaturalScenes


import os
import h5py
import numpy as np
import pandas as pd
import itertools


save_spikes = False
save_stimulus_table = False
save_sweep_events = False


stims_lu = {
    'static_grating': StaticGratings,
    'drifting_grating': DriftingGratings,
    'natural_scene': NaturalScenes
}


def create_data(mouseid, stimulus_type, base_dir='data', rnd_seed=0):
    """Runs the original ecephys stimulus analysis (ecephys_analysis_modules) and saves the data, in order to be
    used for regression testing. Currently only works with NWB1 files.

    :param mouseid:
    :param stimulus_type:
    :param base_dir:
    :param rnd_seed:
    """

    nwb_path = os.path.join(base_dir, '{}.spikes.nwb'.format(mouseid))
    # stimulus_type = 'static_grating'
    np.random.seed(rnd_seed)
    sg = stims_lu[stimulus_type](nwb_path=nwb_path)

    with h5py.File('expected/{}.{}.h5'.format(mouseid, stimulus_type), 'w') as h5:
        h5.attrs['numbercells'] = sg.numbercells
        h5.attrs['rnd_seed'] = rnd_seed

        if save_spikes:
            # record spikes used for analysis
            spikes_grp = h5.create_group('/spikes')
            for specimen_id, spikes in sg.spikes.items():
                cell_grp = spikes_grp.create_group(str(specimen_id))
                cell_grp.create_dataset('spikes', data=spikes)

        if save_stimulus_table:
            # record the stimulus table section
            stim_table_grp = h5.create_group('/stimulus_table')
            for col in sg.stim_table.columns:
                stim_table_grp.create_dataset(col, data=sg.stim_table[col])

            # record the
            spon_stim_table_grp = h5.create_group('/stimulus_table_spontaneous')
            for col in sg.stim_table_sp.columns:
                spon_stim_table_grp.create_dataset(col, data=sg.stim_table_sp[col])

        if save_sweep_events:
            # A table of spike-trains (presentation-ids x unit-ids)
            save_sweep_data(h5, sg.sweep_events)

        # running speed data
        h5.create_dataset('dxcm', data=sg.dxcm)
        h5.create_dataset('dxcm_ts', data=sg.dxcm_ts)
        h5.create_dataset('running_speed', data=sg.running_speed['running_speed'].astype(np.float64))

        mse_grp = h5.create_group('mean_sweep_events')
        mse_grp.create_dataset('specimen_ids', data=[int(i) for i in sg.mean_sweep_events.columns.values])
        mse_grp.create_dataset('presentation_ids', data=[int(i) for i in sg.mean_sweep_events.index.values])
        mse_grp.create_dataset('data', data=sg.mean_sweep_events.values)

        sweep_p_vals = h5.create_group('sweep_p_values')
        sweep_p_vals.create_dataset('specimen_ids', data=[int(i) for i in sg.sweep_p_values.columns.values])
        sweep_p_vals.create_dataset('presentation_ids', data=[int(i) for i in sg.sweep_p_values.index.values])
        sweep_p_vals.create_dataset('data', data=sg.sweep_p_values.values)

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
        'responsive_dg': bool,
        'num_pref_trials_dg': np.uint64,
        'num_pref_trials_ns': np.uint64,
        'responsive_ns': bool,
        'pref_image_ns': np.uint64
        # 'fit_sf_ind_sg': np.uint64
    }

    peak_grp = h5.create_group('peak')
    for col in peak_df.columns:
        dtype = dtype_lu.get(col, np.float64)
        peak_grp.create_dataset(col, data=peak_df[col].astype(dtype))


def filter_nwb(mouseid, base_dir='data', stim_presenations=None, unit_list=None):
    """Used the cull the original nwb file so that integration tests can be run in a reasonable amount of time and the
    files can be saved in github.

    :param mouseid:
    :param base_dir:
    :param stim_presenations:
    :param unit_list:
    :return:
    """
    stim_presentations = stim_presenations or ['Natural Images_5', 'drifting_gratings_2', 'spontaneous',
                                               'static_gratings_6']

    orig_h5 = h5py.File(os.path.join(base_dir, '{}.spikes.nwb'.format(mouseid)), 'r')
    filtered_mouseid = '{}.filtered'.format(mouseid)

    with h5py.File(os.path.join(base_dir, '{}.spikes.nwb'.format(filtered_mouseid)), 'w') as h5:
        orig_h5.copy('/acquisition', h5)
        orig_h5.copy('/general', h5)
        orig_h5.copy('/identifier', h5)
        orig_h5.copy('/nwb_version', h5)
        orig_h5.copy('/session_description', h5)
        orig_h5.copy('/session_start_time', h5)

        presentation_grp = h5.create_group('stimulus/presentation')
        for stim_name, stim_grp in orig_h5['/stimulus/presentation'].items():
            if stim_name in stim_presentations:
                orig_h5.copy(stim_grp, presentation_grp)

        processing_grp_new = h5.create_group('processing')
        for probe_name, probe_grp in orig_h5['processing'].items():
            print(probe_name)
            selected_units = []
            for unit_id, unit_grp in probe_grp['UnitTimes'].items():
                if not isinstance(unit_grp, h5py.Group):
                    continue
                try:
                    ccf_struct = str(unit_grp['ccf_structure'][()], encoding='ascii')
                except TypeError:
                    ccf_struct = unit_grp['ccf_structure'][()]

                if ccf_struct == 'VISp':
                    if unit_list:
                        if int(unit_id) in unit_list:
                            selected_units.append((unit_id, unit_grp))
                    else:
                        selected_units.append((unit_id, unit_grp))

            if selected_units:
                probe_grp_new = processing_grp_new.create_group('{}'.format(probe_name))
                unittimes_grp_new = probe_grp_new.create_group('UnitTimes')
                unit_ids = []
                for unit_id, unit_grp in selected_units:
                    orig_h5.copy(unit_grp, unittimes_grp_new)
                    unit_ids.append(int(unit_id))

                unit_ids = np.sort(unit_ids)
                probe_grp_new.create_dataset('unit_list', data=unit_ids)
                # Needed for Static_gratings_svd
                unittimes_grp_new.create_dataset('unit_list', shape=(0,), dtype=np.float64)

    return filtered_mouseid


if __name__ == '__main__':
    mouseid = 'mouse412792'

    '''
    mouseid = 'mouse412792'
    unit_list = [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181,
                 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                 
                 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 222, 223]
    mouseid = filter_nwb(mouseid, unit_list=unit_list[:20])
    
    '''

    create_data(mouseid=mouseid, stimulus_type='static_grating')
    create_data(mouseid=mouseid, stimulus_type='drifting_grating')
    create_data(mouseid=mouseid, stimulus_type='natural_scene')
