# from ecephys_analysis_modules.common.static_gratings_sdv import StaticGratings
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
import numpy as np
import h5py


def cmp_peak_data(actual_df, expected_h5):
    failed = []
    peak_grp = expected_h5['peak']
    sid_lu = {sid: i for i, sid in enumerate(peak_grp['cell_specimen_id'])}

    actual_df.set_index('cell_specimen_id', inplace=True)
    actual_ids = set(actual_df.index.values.astype(np.uint))
    expected_ids = set(peak_grp['cell_specimen_id'][()])
    if actual_ids != expected_ids:
        print('specimen_ids do not match.')
        return False
    for sid in actual_ids:
        actual_pd = actual_df.loc[str(sid)]
        expected_indx = sid_lu[sid]
        for col in ['pref_ori_sg', 'pref_sf_sg', 'pref_phase_sg', 'num_pref_trials_sg', 'responsive_sg',
                    'g_osi_sg', 'sfdi_sg', 'reliability_sg', 'lifetime_sparseness_sg', 'fit_sf_sg',
                    'fit_sf_ind_sg', 'sf_low_cutoff_sg', 'sf_high_cutoff_sg', 'run_pval_sg', 'run_mod_sg',
                    'run_resp_sg', 'stat_resp_sg', 'lifetime_sparseness_dg']:

            if np.isnan(actual_pd[col]):
                if not np.isnan(peak_grp[col][expected_indx]):
                    failed.append('{}, {}> {} != {}'.format(sid, col, actual_pd[col], peak_grp[col][expected_indx]))
            else:
                if actual_pd[col] != peak_grp[col][expected_indx]:
                    failed.append('{}, {}> {} != {}'.format(sid, col, actual_pd[col], peak_grp[col][expected_indx]))
    if failed:
        print(failed)
        return False

    return True


def cmp_mean_sweeps(actual_df, expected_h5):
    failed = []
    mse_grp = expected_h5['mean_sweep_events']
    actual_ids = actual_df.columns.values.astype(np.uint64)
    expected_ids = mse_grp['specimen_ids'][()]
    if set(actual_ids) != set(expected_ids):
        print('specimen_ids do not match.')
        return False

    sid_lu = {sid: i for i, sid in enumerate(mse_grp['specimen_ids'])}
    for sid in actual_ids:
        expected_indx = sid_lu[sid]
        expected_vals = mse_grp['data'][:, expected_indx]
        actual_vals = actual_df[str(sid)].values
        if not np.all(expected_vals == actual_vals):
            failed.append(sid)

    if failed:
        print('failed specimen ids: {}'.format(failed))
        return False
    return True


def cmp_p_sweeps(actual_df, expected_h5):
    failed = []
    spv_grp = expected_h5['sweep_p_values']
    actual_ids = actual_df.columns.values.astype(np.uint64)
    expected_ids = spv_grp['specimen_ids'][()]
    if set(actual_ids) != set(expected_ids):
        print('specimen_ids do not match.')
        return False

    sid_lu = {sid: i for i, sid in enumerate(spv_grp['specimen_ids'])}
    for sid in actual_ids:
        expected_indx = sid_lu[sid]
        expected_vals = spv_grp['data'][:, expected_indx]
        actual_vals = actual_df[str(sid)].values
        if not np.allclose(expected_vals, actual_vals, atol=1.0e-3):
            for i in range(6000):
                if np.abs(expected_vals[i] - actual_vals[i]) > 1.0e-3:
                    print(expected_vals[i], actual_vals[i])

            failed.append(sid)

    if failed:
        print('failed specimen ids: {}'.format(failed))
        return False
    return True


def cmp_sweep_events(actual_df, expected_h5, id_map=None):
    sw_grp = expected_h5['/sweep_events']
    sid_lu = {sid: i for i, sid in enumerate(sw_grp['specimen_ids'])}
    failed = []
    id_map = id_map or {}
    for sid in sw_grp['specimen_ids']: #actual_df.columns.values:
        print(sid)
        act_sid = id_map[sid]
        exp_sid = str(sid)
        # print(sid, id_map[sid])
        #print(actual_df[731])
        #exit()
        #for pid in actual_df.index.values:
        for pid in range(len(actual_df)):
            # actual_events = actual_df[id_map[sid]].iloc[pid]
            actual_events = actual_df[act_sid].iloc[pid]

            exp_indx = sid_lu[int(sid)]
            exp_rref = sw_grp['events_table'][pid, exp_indx]
            expected_events = sw_grp[exp_sid]['data'][exp_rref] if bool(exp_rref) else []
            if not np.allclose(actual_events, expected_events):
                failed.append('sid={},pid={}'.format(sid, pid))

    if failed:
        print(failed)
        return False

    return True


def cmp_peak(actual_df, expected_h5):
    peak_grp = expected_h5['/peak']
    actual_df = actual_df.set_index('cell_specimen_id')
    sid_lu = {sid: i for i, sid in enumerate(peak_grp['cell_specimen_id'])}

    failed = False
    for sid in actual_df.index.values:
        actual_features = actual_df.loc[sid]
        expected_indx = sid_lu[int(sid)]
        for col in actual_df.columns.values:
            actual_val = actual_features[col]
            expected_val = peak_grp[col][expected_indx]

            if np.isnan(actual_val) != np.isnan(expected_val):
                print(sid)
                print(col, actual_val, expected_val)
                failed = True
            elif not np.isnan(actual_val) and actual_val != expected_val:
                print(sid)
                print(col, actual_val, expected_val)
                failed = True

    return not failed


def cmp_spikes(spikes_dict, expected_h5, id_map=None):
    failed = []
    spikes_grp = expected_h5['spikes']
    id_map = id_map or {}
    for specimen_id in spikes_grp.keys():
        expected_ts = spikes_grp[specimen_id]['spikes'][()]
        # print(specimen_id)
        actual_ts = spikes_dict[id_map.get(int(specimen_id), specimen_id)]
        if not np.allclose(actual_ts, expected_ts):
            failed.append(specimen_id)
        #print(expected_ts)
        #exit()
    if failed:
        print('failed specimen_ids: {}'.format(failed))
        return False

    return True
    #print(list(spikes_grp.keys()))
    #exit()



def test_sg_data(spikes_file, expected_file):
    from allensdk.brain_observatory.ecephys.static_gratings import StaticGratings
    expected_h5 = h5py.File(expected_file, 'r')
    if 'rnd_seed' in expected_h5.attrs:
        np.random.seed(expected_h5.attrs['rnd_seed'])

    ecephys_session = EcephysSession.from_nwb1_path(spikes_file)
    units = ecephys_session.units
    units = units[(units['location'] == 'probeC') & (units['structure_acronym'] == 'VISp')]
    id_map = {loc_id: unit_id for loc_id, unit_id in zip(units['local_index_unit'], units.index.values)}
    print(id_map)

    #print(ecephys_session.units.columns)
    #print(id_map)
    # print(units['local_index_unit'])
    #exit()

    sg = StaticGratings(spikes_file)

    # print(np.allclose(expected_h5['dxcm'][()], sg.dxcm))
    # print(np.allclose(expected_h5['dxcm_ts'][()], sg.dxtime))
    # print(cmp_spikes(sg.spikes, expected_h5, id_map))
    # print(expected_h5.attrs['numbercells'] == sg.numbercells)
    # print(sg.stim_table.columns)
    # print(sg.stim_table_spontaneous)
    #sweep_events = sg.sweep_events
    #print(sweep_events)
    #print(cmp_sweep_events(sweep_events, expected_h5, id_map))
    print(sg.running_speed)
    exit()
    

    print(np.allclose(sg.dxcm, expected_h5['dxcm'][()]))

    assert(cmp_peak(sg.peak.copy(), expected_h5))
    assert(cmp_p_sweeps(sg.sweep_p_values.copy(), expected_h5))
    assert(cmp_peak_data(sg.peak.copy(), expected_h5))
    assert(cmp_mean_sweeps(sg.mean_sweep_events.copy(), expected_h5))
    assert(cmp_sweep_events(sg.sweep_events.copy(), expected_h5))


if __name__ == '__main__':
    test_sg_data('data/mouse412792.spikes.nwb', expected_file='expected/mouse412792.static_grating.h5')
