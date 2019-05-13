import numpy as np
import h5py

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.drifting_gratings import DriftingGratings
from allensdk.brain_observatory.ecephys.natural_scenes import NaturalScenes
from allensdk.brain_observatory.ecephys.ecephys_api import EcephysNwb1Adaptor


def cmp_spikes(spikes_dict, expected_h5, id_map=None):
    if id_map is not None:
        spikes_dict = {id_map[k]: v for k, v in spikes_dict.items()}

    failed = []
    spikes_grp = expected_h5['spikes']
    for specimen_id in spikes_grp.keys():
        expected_ts = spikes_grp[specimen_id]['spikes'][()]
        actual_ts = spikes_dict[int(specimen_id)]
        if not np.allclose(actual_ts, expected_ts):
            failed.append(specimen_id)

    if failed:
        print('failed specimen_ids: {}'.format(failed))
        return False

    return True


def cmp_sweep_events(actual_df, expected_h5, id_map=None, sampled=None):
    # TODO: This can take a long time so add option to test against a random sample
    if id_map is not None:
        actual_df = actual_df.rename(index=str, columns=id_map)

    sw_grp = expected_h5['/sweep_events']
    sid_lu = {sid: i for i, sid in enumerate(sw_grp['specimen_ids'])}
    failed = []

    for sid in sw_grp['specimen_ids']:  # actual_df.columns.values:
        exp_sid = str(sid)
        for pid in range(len(actual_df)):
            actual_events = actual_df[sid].iloc[pid]

            exp_indx = sid_lu[int(sid)]
            exp_rref = sw_grp['events_table'][pid, exp_indx]
            expected_events = sw_grp[exp_sid]['data'][exp_rref] if bool(exp_rref) else []
            if not np.allclose(actual_events, expected_events):
                failed.append('sid={},pid={}'.format(sid, pid))

    if failed:
        print(failed)
        return False

    return True


def cmp_running_speed(stim_analysis, expected_h5):
    running_speed = stim_analysis.running_speed['running_speed'].values
    rs_actual = running_speed[~np.isnan(running_speed)]
    running_speed = expected_h5['running_speed'][()]
    rs_expected = running_speed[~np.isnan(running_speed)]
    return np.allclose(rs_actual, rs_expected, atol=1.0e-5)


def cmp_mean_sweeps(actual_df, expected_h5, id_map=None):
    if id_map is not None:
        actual_df = actual_df.rename(columns=id_map)

    failed = []
    mse_grp = expected_h5['mean_sweep_events']
    actual_ids = actual_df.columns.values.astype(np.uint64)
    expected_ids = mse_grp['specimen_ids'][()]
    if set(actual_ids) != set(expected_ids):
        print('specimen_ids do not match.')
        return False

    sid_lu = {id_map.get(sid, sid): i for i, sid in enumerate(mse_grp['specimen_ids'])}
    for sid in actual_ids:
        expected_indx = sid_lu[sid]
        expected_vals = mse_grp['data'][:, expected_indx]
        actual_vals = actual_df[sid].values  # actual_df[str(sid)].values
        if not np.all(expected_vals == actual_vals):
            failed.append(sid)

    if failed:
        print('failed specimen ids: {}'.format(failed))
        return False
    return True


def cmp_p_sweeps(actual_df, expected_h5, id_map=None):
    if id_map is not None:
        actual_df = actual_df.rename(index=str, columns=id_map)

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
        actual_vals = actual_df[sid].values  # actual_df[str(sid)].values
        if not np.allclose(expected_vals, actual_vals, atol=1.0e-3):
            for i in range(6000):
                if np.abs(expected_vals[i] - actual_vals[i]) > 1.0e-3:
                    print(expected_vals[i], actual_vals[i])

            failed.append(sid)

    if failed:
        print('failed specimen ids: {}'.format(failed))
        return False
    return True

def cmp_peak_data(actual_df, expected_h5, id_map=None):
    failed = []
    peak_grp = expected_h5['peak']
    sid_lu = {sid: i for i, sid in enumerate(peak_grp['cell_specimen_id'])}

    actual_df.set_index('cell_specimen_id', inplace=True)
    if id_map is not None:
        actual_df = actual_df.rename(index=id_map)
    actual_ids = set(actual_df.index.values.astype(np.uint))
    expected_ids = set(peak_grp['cell_specimen_id'][()])

    if actual_ids != expected_ids:
        print('specimen_ids do not match.')
        return False

    # print(peak_grp['cell_specimen_id'])
    # expected_specimen_ids = peak_grp['cell_specimen_id'].values
    # expected_sort_order = np.argsort(peak_grp['cell_specimen_id'][()])
    # print(expected_sort_order)
    # print(peak_grp['cell_specimen_id'][()])
    actual_df = actual_df.reindex(index=peak_grp['cell_specimen_id'][()])
    #print(actual_df)
    #exit()

    assert(set(actual_df.columns) == set([k for k in peak_grp.keys() if k != 'cell_specimen_id']))
    #print(set([k for k in peak_grp.keys() if k != 'cell_specimen_id']))
    #exit()


    # TODO: Get column names from stimulus_analysis
    #for col in ['pref_ori_dg', 'pref_tf_dg', 'num_pref_trials_dg', 'responsive_dg', 'g_osi_dg', 'g_dsi_dg', 'tfdi_dg',
    #            'reliability_dg', 'lifetime_sparseness_dg', 'fit_tf_dg', 'fit_tf_ind_dg', 'tf_low_cutoff_dg',
    #            'tf_high_cutoff_dg', 'run_pval_dg', 'run_resp_dg', 'stat_resp_dg', 'run_mod_dg',
    #            'peak_blank_dg', 'all_blank_dg']:
    for col in actual_df.columns:

        # TODO: Need reorder the actual column so the specimen_ids allways match up with expected
        #print(actual_df[col].values[expected_sort_order])
        #exit()
        #print(col)
        #print(actual_df[col].dtype)
        #print(actual_df[col].values)
        #print(peak_grp[col][()])
        #print(actual_df)
        #print(actual_df[[col]])
        if not np.allclose(actual_df[col].values, peak_grp[col][()], equal_nan=True):
            failed.append(col)
            #print(col)
            #print(actual_df[col].values)
            #exit()

    if failed:
        print(failed)
        return False

    return True


def test_stimulus_data(spikes_file, expected_file, stim_analysis_class):
    expected_h5 = h5py.File(expected_file, 'r')
    if 'rnd_seed' in expected_h5.attrs:
        np.random.seed(expected_h5.attrs['rnd_seed'])

    ecephys_session = EcephysSession.from_nwb1_path(spikes_file)
    stim_table = stim_analysis_class(spikes_file)

    if isinstance(stim_table.ecephys_session.api, EcephysNwb1Adaptor):
        units = ecephys_session.units
        units = units[(units['location'] == 'probeC') & (units['structure_acronym'] == 'VISp')]
        id_map = {unit_id: loc_id for loc_id, unit_id in zip(units['local_index_unit'], units.index.values)}
    else:
        id_map = None

    assert(np.allclose(expected_h5['dxcm'][()], stim_table.dxcm))
    assert(np.allclose(expected_h5['dxcm_ts'][()], stim_table.dxtime))
    assert(expected_h5.attrs['numbercells'] == stim_table.numbercells)
    print(stim_table.stim_table.columns)
    exit()

    #print(dg.stim_table.columns)
    #print(dg.stim_table[['start_time', 'SF']])
    #print(expected_h5['stimulus_table'])

    if 'spikes' in expected_h5:
        assert(cmp_spikes(stim_table.spikes, expected_h5, id_map))

    #if 'sweep_events' in expected_h5:
    #    assert(cmp_sweep_events(stim_table.sweep_events.copy(), expected_h5, id_map))

    #print(dg.running_speed[:10])
    #print(expected_h5['running_speed'][:10])
    assert(cmp_running_speed(stim_table, expected_h5))
    assert(cmp_mean_sweeps(stim_table.mean_sweep_events.copy(), expected_h5, id_map))
    #print(dg.sweep_p_values[739])
    #print(id_map)
    assert(cmp_p_sweeps(stim_table.sweep_p_values.copy(), expected_h5, id_map))
    assert(cmp_peak_data(stim_table.peak.copy(), expected_h5, id_map))


if __name__ == '__main__':
    # mouseid = 'mouse412792.filtered'
    mouseid = 'mouse412792'
    # test_sg_data('data/{}.spikes.nwb'.format(mouseid), expected_file='expected/{}.drifting_grating.h5'.format(mouseid))
    test_stimulus_data('data/{}.spikes.nwb'.format(mouseid), expected_file='expected/{}.natural_scene.h5'.format(mouseid), stim_analysis_class=NaturalScenes)
