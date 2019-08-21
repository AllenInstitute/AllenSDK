import pytest
import numpy as np
import pandas as pd
from mock import MagicMock

from allensdk.brain_observatory.ecephys.stimulus_analysis import stimulus_analysis as sa
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


@pytest.fixture
def ecephys_session():
    ecephys_ses = MagicMock(spec=EcephysSession)
    units_df = pd.DataFrame({'unit_id': np.arange(20)})
    units_df = units_df.set_index('unit_id')
    ecephys_ses.units = units_df
    ecephys_ses.spike_times = {uid: np.linspace(0, 1.0, 5) for uid in np.arange(20)}
    ecephys_ses.running_speed = pd.DataFrame({'start_time': np.linspace(0, 100.0, 20), 'velocity': [0.001]*20})
    return ecephys_ses


@pytest.mark.skip()
# @patch('allensdk.brain_observatory.ecephys.ecephys_session.EcephysSession')
def test_stimulus_analysis(ecephys_session):
    sa_obj = sa.StimulusAnalysis(ecephys_session)
    assert(np.all(np.sort(sa_obj.unit_ids) == np.arange(20)))
    assert(sa_obj.numbercells == 20)
    assert(len(sa_obj.spikes) == 20)
    assert(np.all(sa_obj.spikes[0] == np.linspace(0, 1.0, 5)))
    assert(np.allclose(sa_obj.dxtime, np.linspace(0, 100.0, 20)))
    assert(np.allclose(sa_obj.dxcm, [0.001]*20))


@pytest.mark.skip()
@pytest.mark.parametrize('sweeps,win_beg,win_end,expected,as_array',
                         [
                             ([[0.82764702, 0.83624702, 1.09211374], [0.34899642, 0.49176312, 0.71626316], [0.1549371], [0.89921008, 1.07917679]], 30, 40, -0.6204065787016709, False),
                             ([[0.82764702, 0.83624702, 1.09211374], [0.34899642, 0.49176312, 0.71626316], [0.1549371], [0.89921008, 1.07917679]], 30, 40, -0.6204065787016709, True),
                             ([[0.82764702, 0.83624702, 1.09211374, 1.112345]], 30, 40, np.nan, True)
                          ]
                         )
def test_get_reliability(sweeps, win_beg, win_end, expected, as_array):
    sweeps = np.array([np.array(s) for s in sweeps]) if as_array else sweeps
    assert(np.isclose(sa.get_reliability(unit_sweeps=sweeps, window_beg=win_beg, window_end=win_end), expected,
                      equal_nan=True))


@pytest.mark.parametrize('spikes,sampling_freq,sweep_length,expected',
                         [
                             (np.array([0.82764702, 0.83624702, 1.09211374]), 10, 1.5, [0.0, 0.0, 0.0, 0.0, 0.000133830, 0.004431861, 0.05412495, 0.2464033072, 0.45293459677, 0.4839428913,
                                                                                        0.452934596, 0.2464033072, 0.054124958, 0.00443186162, 0.0001338306]),
                             (np.array([]), 10, 1.5, np.zeros(15))
                         ])
def test_get_fr(spikes, sampling_freq, sweep_length, expected):
    frs = sa.get_fr(spikes, num_timestep_second=sampling_freq, sweep_length=sweep_length)
    assert(len(frs) == int(sampling_freq*sweep_length))
    assert(np.allclose(frs, expected))


@pytest.mark.skip()
@pytest.mark.parametrize('responses,expected',
                         [
                             (np.array([2.24, 3.6, 0.8, 2.4, 3.52, 5.68, 8.96, 0.8, 0.8, 2.64, 0.96, 2.64, 0.16, 2.16, 0.0, 1.76, 2.88, 3.12, 0.0, 1.44]), np.array([0.4625097])),
                             (np.array([[2.24, 3.60, 3.12, 0.00], [1.76, 4.32, 3.04, 1.68], [2.75, 4.75, 0.33, 1.25], [2.16, 3.44, 4.96, 6.24], [1.68, 4.00, 7.02, 1.42]]),
                              np.array([0.0397182, 0.01731439, 0.33262806, 0.63161215]))
                         ])
def test_get_lifetime_sparseness(responses, expected):
    assert(np.allclose(sa.get_lifetime_sparseness(responses), expected))


@pytest.mark.skip()
@pytest.mark.parametrize('responses,ori_vals,expected',
                         [
                             (np.array([4.32, 6.0, 3.68, 5.04347826, 2.12244898, 3.67346939]), np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]), 0.1439412792058277)
                         ])
def test_get_osi(responses, ori_vals, expected):
    assert(np.isclose(sa.get_osi(responses, ori_vals), expected))


@pytest.mark.skip()
@pytest.mark.parametrize('mean_sweep_runs,mean_sweep_stats,expected',
                         [
                             (np.array([0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 4.0, 8.0, 4.0, 0.0, 8.0, 8.0, 24.0, 16.0, 16.0, 12.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]),
                              np.array([4.0, 0.0, 0.0, 12.0, 4.0, 4.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 8.0, 0.0, 8.0, 4.0, 4.0, 0.0, 0.0, 4.0]),
                              np.array([0.3892556112603236, 0.2688172043010753, 4.428571428571429, 3.238095238095238])),
                             (np.array([0.0]), np.array([4.0, 0.0, 0.0, 12.0]), np.full(4, np.nan)),
                             (np.array([4.0, 0.0, 0.0, 12.0]), np.array([1.0]), np.full(4, np.nan))
                         ])
def test_get_running_modulation(mean_sweep_runs, mean_sweep_stats, expected):
    assert(np.allclose(sa.get_running_modulation(mean_sweep_runs, mean_sweep_stats), expected, equal_nan=True))


if __name__ == '__main__':
    #unit_sweeps = [[0.82764702, 0.83624702, 1.09211374], [0.34899642, 0.49176312, 0.71626316], [0.1549371], [0.89921008, 1.07917679]]
    #test_get_reliability(np.array([np.array(l) for l in unit_sweeps]), -0.6204065787016709)  # 83758691)
    #unit_sweeps = [[0.82764702, 0.83624702, 1.09211374, 1.112345]]
    #test_get_reliability(unit_sweeps, np.nan)

    #test_get_fr(np.array([0.82764702, 0.83624702, 1.09211374]), sampling_freq=10, sweep_length=1.5, expected=[0.0, 0.0, 0.0, 0.0, 0.00013383062461474175, 0.0044318616200312655, 0.05412495804531915, 0.24640330727663198, 0.45293459677680215, 0.48394289131320145, 0.45293459677680215, 0.24640330727663198, 0.05412495804531915, 0.0044318616200312655, 0.00013383062461474175])
    #test_get_fr(np.array([]), sampling_freq=10, sweep_length=1.5, expected=np.zeros(15))
    #test_get_fr(np.array([0.0]), sampling_freq=10, sweep_length=1.5, expected=[0.6409149150126985, 0.2959625730773051, 0.05842298904073567, 0.004565692244646007, 0.00013383062461474175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #test_get_lifetime_sparseness(np.array([2.24, 3.6, 0.8, 2.4, 3.52, 5.68, 8.96, 0.8, 0.8, 2.64, 0.96, 2.64, 0.16, 2.16, 0.0, 1.76, 2.88, 3.12, 0.0, 1.44]), expected=np.array([0.4625097]))
    #test_get_lifetime_sparseness(np.array([[2.24, 3.60, 3.12, 0.00],
    #                                       [1.76, 4.32, 3.04, 1.68],
    #                                       [2.75, 4.75, 0.33, 1.25],
    #                                       [2.16, 3.44, 4.96, 6.24],
    #                                       [1.68, 4.00, 7.02, 1.42]]), expected=np.array([0.0397182, 0.01731439, 0.33262806, 0.63161215]))

    # test_get_osi(np.array([4.32, 6.0, 3.68, 5.04347826, 2.12244898, 3.67346939]), np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]), 0.1439412792058277)

    test_get_running_modulation(np.array([0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 4.0, 8.0, 4.0, 0.0, 8.0, 8.0, 24.0, 16.0, 16.0, 12.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]),
                                np.array([4.0, 0.0, 0.0, 12.0, 4.0, 4.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 8.0, 0.0, 8.0, 4.0, 4.0, 0.0, 0.0, 4.0]),
                                np.array([0.3892556112603236, 0.2688172043010753, 4.428571428571429, 3.238095238095238]))

    pass
