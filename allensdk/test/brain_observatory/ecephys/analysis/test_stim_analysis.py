import os
import pytest
import numpy as np
import h5py

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.static_gratings import StaticGratings
#from allensdk.brain_observatory.ecephys.drifting_gratings import DriftingGratings
#from allensdk.brain_observatory.ecephys.natural_scenes import NaturalScenes
from allensdk.brain_observatory.ecephys.ecephys_session_api.ecephys_nwb1_session_api import EcephysNwb1Api


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
            for i in range(len(expected_vals)):
                if np.abs(expected_vals[i] - actual_vals[i]) > 1.0e-3:
                    print(expected_vals[i], actual_vals[i])

            failed.append(sid)

    if failed:
        print('failed specimen ids: {}'.format(failed))
        return False
    return True

def cmp_peak_data(actual_df, expected_h5, id_map=None):
    """Compare the peak table.

    :param actual_df:
    :param expected_h5:
    :param id_map:
    """
    failed = []
    peak_grp = expected_h5['peak']
    # sid_lu = {sid: i for i, sid in enumerate(peak_grp['cell_specimen_id'])}

    actual_df.set_index('cell_specimen_id', inplace=True)
    if id_map is not None:
        actual_df = actual_df.rename(index=id_map)
    actual_ids = set(actual_df.index.values.astype(np.uint))
    expected_ids = set(peak_grp['cell_specimen_id'][()])

    if actual_ids != expected_ids:
        print('specimen_ids do not match.')
        return False

    # make sure not missing any peak columns
    assert(set(actual_df.columns) == set([k for k in peak_grp.keys() if k != 'cell_specimen_id']))

    actual_df = actual_df.reindex(index=peak_grp['cell_specimen_id'][()]) # make sure cell_id order is the same
    for col in actual_df.columns:
        if not np.allclose(actual_df[col].values, peak_grp[col][()], equal_nan=True):
            failed.append(col)

    if failed:
        print(failed)
        return False

    return True


@pytest.mark.parametrize('spikes_file,expected_file,stim_analysis_class,nwb_version',
                         [('data/mouse412792.filtered.spikes.nwb', 'expected/mouse412792.filtered.static_grating.h5', StaticGratings, 1),
                          #('data/mouse412792.filtered.spikes.nwb', 'expected/mouse412792.filtered.drifting_grating.h5', DriftingGratings, 1),
                          #('data/mouse412792.filtered.spikes.nwb', 'expected/mouse412792.filtered.natural_scene.h5', NaturalScenes, 1)
                          ])
def test_stimulus_data(spikes_file, expected_file, stim_analysis_class, nwb_version):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    spikes_file = os.path.join(file_dir, spikes_file)
    expected_file = os.path.join(file_dir, expected_file)
    expected_h5 = h5py.File(expected_file, 'r')
    if 'rnd_seed' in expected_h5.attrs:
        np.random.seed(expected_h5.attrs['rnd_seed'])

    ecephys_session = EcephysSession.from_nwb_path(path=spikes_file, nwb_version=nwb_version)
    sa = stim_analysis_class(ecephys_session)

    if isinstance(sa.ecephys_session.api, EcephysNwb1Api):
        units = ecephys_session.units
        units = units[(units['location'] == 'probeC') & (units['structure_acronym'] == 'VISp')]
        id_map = {unit_id: loc_id for loc_id, unit_id in zip(units['local_index_unit'], units.index.values)}
    else:
        id_map = None

    assert(np.allclose(expected_h5['dxcm'][()], sa.dxcm))
    assert(np.allclose(expected_h5['dxcm_ts'][()], sa.dxtime))
    assert(expected_h5.attrs['numbercells'] == sa.numbercells)

    if 'spikes' in expected_h5:
        assert(cmp_spikes(sa.spikes, expected_h5, id_map))

    if 'sweep_events' in expected_h5:
        assert(cmp_sweep_events(sa.sweep_events.copy(), expected_h5, id_map))

    assert(cmp_running_speed(sa, expected_h5))
    assert(cmp_mean_sweeps(sa.mean_sweep_events.copy(), expected_h5, id_map))
    assert(cmp_p_sweeps(sa.sweep_p_values.copy(), expected_h5, id_map))
    assert(cmp_peak_data(sa.peak.copy(), expected_h5, id_map))


if __name__ == '__main__':
    mouseid = 'mouse412792.filtered'
    # mouseid = 'mouse412792'
    test_stimulus_data('data/{}.spikes.nwb'.format(mouseid), expected_file='expected/{}.static_grating.h5'.format(mouseid), stim_analysis_class=StaticGratings, nwb_version=1)
    #test_stimulus_data('data/{}.spikes.nwb'.format(mouseid), expected_file='expected/{}.drifting_grating.h5'.format(mouseid), stim_analysis_class=DriftingGratings)
    #test_stimulus_data('data/{}.spikes.nwb'.format(mouseid), expected_file='expected/{}.natural_scene.h5'.format(mouseid), stim_analysis_class=NaturalScenes)