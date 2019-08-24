import pytest
import numpy as np
import pandas as pd
from mock import MagicMock

from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.ecephys.stimulus_analysis import stimulus_analysis as sa
from allensdk.brain_observatory.ecephys.stimulus_analysis.stimulus_analysis import StimulusAnalysis, running_modulation, lifetime_sparseness, fano_factor, overall_firing_rate
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



def mock_ecephys_api():
    class EcephysSpikeTimesApi(EcephysSessionApi):
        def get_spike_times(self):
            return {
                0: np.array([1, 2, 3, 4]),
                1: np.array([2.5]),
                2: np.array([1.01, 1.03, 1.02]),
                3: np.array([]),
                4: np.array([0.01, 1.7, 2.13, 3.19, 4.25]),
                5: np.array([1.5, 3.0, 4.5])
            }

        def get_channels(self):
            return pd.DataFrame({
                'local_index': [0, 1, 2],
                'probe_horizontal_position': [5, 10, 15],
                'probe_id': [0, 0, 1],
                'probe_vertical_position': [10, 22, 33],
                'valid_data': [False, True, True]
            }, index=pd.Index(name='channel_id', data=[0, 1, 2]))

        def get_units(self):
            udf = pd.DataFrame({
                'firing_rate': np.linspace(1, 3, 4),
                'isi_violations': [40, 0.5, 0.1, 0.2],
                'local_index': [0, 0, 1, 1],
                'peak_channel_id': [0, 2, 1, 1],
                'quality': ['good', 'good', 'good', 'bad'],
                #'snr': [0.1, 1.4, 10.0, 0.3]
            }, index=pd.Index(name='unit_id', data=np.arange(4)[::-1]))
            return udf

        def get_probes(self):
            return pd.DataFrame({
                'description': ['probeA', 'probeB'],
                'location': ['VISp', 'VISam'],
                'sampling_rate': [30000.0, 30000.0]
            }, index=pd.Index(name='id', data=[0, 1]))

        def get_stimulus_presentations(self):
            return pd.DataFrame({
                'start_time': np.linspace(0.0, 4.5, 10, endpoint=True),
                'stop_time': np.linspace(0.5, 5.0, 10, endpoint=True),
                'stimulus_name': ['spontaneous'] + ['s0']*6 + ['spontaneous'] + ['s1']*2,
                'stimulus_block': [0] + [1]*6 + [0] + [2]*2,
                'duration': 0.5,
                #'TF': np.empty(4) * np.nan,
                #'SF': np.empty(4) * np.nan,
                #'Ori': np.empty(4) * np.nan,
                #'Contrast': np.empty(4) * np.nan,
                #'Pos_x': np.empty(4) * np.nan,
                #'Pos_y': np.empty(4) * np.nan,
                'stimulus_index': [0] + [1]*6 + [0] + [2]*2,
                'conditions': [0, 0, 0, 0, 1, 1, 1, 0, 1, 1]  # generic stimulus condition
                #'conditions': np.arange(10)
                #'Color': np.arange(4) * 5.5,
                #'Image': np.empty(4) * np.nan,
                #'Phase': np.linspace(0, 180, 4),
            }, index=pd.Index(name='id', data=np.arange(10)))

        def get_running_speed(self):
            return pd.DataFrame({
                "start_time": np.linspace(0.0, 9.9, 100),
                "end_time": np.linspace(0.1, 10.0, 100),
                "velocity": np.linspace(-0.1, 11.0, 100)
            })

    return EcephysSpikeTimesApi()

def test_unit_ids():
    """"""
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session)
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

def test_unit_ids_filter_by_id():
    """"""
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, filter=[3, 2, 1])
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'unit_id': [3, 0]})
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

    # TODO: Write test that will fail if bad ids are used

def test_unit_ids_filtered():
    #
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'VISp'})
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'VISp', 'quality': 'good'})
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

    # Bad
    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'pSIV'})
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

def test_filtered_unit_ids_nomatches():
    session = EcephysSession(api=mock_ecephys_api())
    #print(session.units)
    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'invalid'})
    print(stim_analysis.unit_ids)
    print(stim_analysis.unit_count)

def test_stim_table():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='a')
    print(stim_analysis.stim_table)
    print(len(stim_analysis.stim_table))
    print(stim_analysis.total_presentations)


    # TODO: Write test that will fail with bad stimulus_key
    #stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='b')
    #print(stim_analysis.stim_table)

    # TODO: Write test to check sponateous stim table
    # print(stim_analysis.stim_table_spontaneous)

def test_conditionwise_psth():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5, psth_resultion=0.02)
    print(stim_analysis.conditionwise_psth.shape)
    print(stim_analysis.conditionwise_psth)

    # TODO: Write special case where each stimulus_condition_id is unique
    #print(stim_analysis.conditionwise_psth[{'unit_id': 3}])

def test_conditionwise_statistics():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    print(stim_analysis.conditionwise_statistics)

def test_presentationwise_spike_times():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    print(stim_analysis.presentationwise_spike_times)

def test_presentationwise_statistics():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5)
    print(stim_analysis.presentationwise_statistics)

def test_stimulus_conditions():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5)
    print(stim_analysis.stimulus_conditions)

def test_running_speed():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    print(stim_analysis.running_speed)


def test_spikes():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    print(stim_analysis.spikes)

def test_spikes_filtered():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', filter=[0, 2])
    print(stim_analysis.spikes)

def test_stim_table_spontaneous():
    # By default table should be empty because non of the stimulus are above the duration threshold
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    print(stim_analysis.stim_table_spontaneous)

    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', spontaneous_threshold=0.49)
    print(stim_analysis.stim_table_spontaneous)

def test_get_preferred_condition():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    print(stim_analysis._get_preferred_condition(3))

"""
def test_running_modulation():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5, threshold=0.01)
    print(stim_analysis.get_running_modulation(1, stim_analysis._get_preferred_condition(1)))
"""

def test_running_modulation(spike_counts, running_speeds, speed_threshold):
    print(running_modulation(spike_counts, running_speeds, speed_threshold))

def test_lifetime_sparseness(responses):
    print(lifetime_sparseness(responses))

def test_fano_factor(spike_counts):
    print(fano_factor(spike_counts))

def test_get_time_to_peak():
    session = EcephysSession(api=mock_ecephys_api())
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5)
    print(stim_analysis._get_time_to_peak(1, stim_analysis._get_preferred_condition(1)))

def test_overall_firing_rate(start_times, stop_times, spike_times):
    print(overall_firing_rate(start_times, stop_times, spike_times))

## test_unit_ids()
## test_unit_ids_filter_by_id()
## test_unit_ids_filtered()
## test_stim_table()
## test_conditionwise_psth()
## test_conditionwise_statistics()
## test_presentationwise_spike_times()
## test_presentationwise_statistics()
## test_stimulus_conditions()
## test_running_speed()
## test_spikes()
## test_spikes_filtered()
## test_stim_table_spontaneous()
## test_get_preferred_condition()

## Doesn't require MockApi
## test_running_modulation(spike_counts=np.zeros(10), running_speeds=np.zeros(1), speed_threshold=1.0)  # Input error, should return nan
## test_running_modulation(spike_counts=np.zeros(5), running_speeds=np.full(5, 2.0), speed_threshold=1.0)  # returns Nan, always running
## test_running_modulation(spike_counts=np.zeros(5), running_speeds=np.full(5, 2.0), speed_threshold=2.1) # returns Nan, always stationary
## test_running_modulation(spike_counts=np.zeros(5), running_speeds=np.array([0.0, 0.0, 2.0, 2.5, 3.0]), speed_threshold=1.0) # No firing, return Nans
## test_running_modulation(spike_counts=np.ones(5), running_speeds=np.array([0.0, 0.0, 2.0, 2.0, 2.0]), speed_threshold=1.0)  # always the same fr, pval is Nan but run_mod is 0.0
## test_running_modulation(spike_counts=np.array([3.0, 3.0, 1.5, 1.5, 0.9]), running_speeds=np.array([0.0, 0.0, 2.0, 2.5, 3.0]), speed_threshold=1.0) # (0.013559949584378913, -0.5666666666666667)
## test_running_modulation(spike_counts=np.array([3.0, 3.0, 1.5, 1.5, 1.5]), running_speeds=np.array([0.0, 0.0, 2.0, 2.5, 3.0]), speed_threshold=1.0)  # (0.0, -0.5)
## test_running_modulation(spike_counts=np.array([3.0, 3.0, 1.5, 5.5, 2.5]), running_speeds=np.array([0.0, 0.0, 2.0, 2.5, 3.0]), speed_threshold=1.0)  # (0.9024099927051468, 0.052631578947368376)
## test_running_modulation(spike_counts=np.array([0.0, 0.0, 4.0, 4.0]), running_speeds=np.array([0.0, 0.0, 2.5, 3.0]), speed_threshold=1.0)  # (0.0, 1.0)

## Doesn't require MockApi
## test_lifetime_sparseness(np.array([1.0]))  # returns nan
## test_lifetime_sparseness(np.full(20, 3.2))  # always the same, sparsness is 0.0
## test_lifetime_sparseness(np.array([10.0, 0.0, 0.0, 0.0, 0.0]))  # spareness should be 1.0
## test_lifetime_sparseness(np.array(np.array([2.24, 3.6, 0.8, 2.4, 3.52, 5.68, 8.96, 0.0])))  # 0.43500091849856115

## Doesn't require MockApi
## test_fano_factor(np.zeros(10))  # mean 0.0 leads to Nan
## test_fano_factor(np.array([-1.5, 1.5]))  # mean 0
## test_fano_factor(np.ones(5))  # no variance
## test_fano_factor(np.array([1.2, 20.0, 0.0, 36.2, 0.6]))  # 17.921379310344832, High variance
## test_fano_factor(np.array([5.1, 5.3, 5.2, 5.1, 5.2]))  # 0.0010810810810810846, low variance


# test_get_time_to_peak()

## Doesn't require MockApi
## test_overall_firing_rate(np.array([0.0]), np.array([0.0]), np.linspace(0, 10.0, 10))  # nan, total time 0.0
## test_overall_firing_rate(np.arange(1.0, 3.0), np.arange(0.0, 2.0), np.linspace(0, 10.0, 10))  # nan, total_time negative
## test_overall_firing_rate(np.arange(1.0, 4.0), np.arange(0.0, 2.0), np.linspace(0, 10.0, 10))  # nan, time lengths don't match
## test_overall_firing_rate(np.array([0.0]), np.array([1.0]), np.linspace(0, 10.0, 100))  # 10.0 Hz
## test_overall_firing_rate(np.array([0.0, 9.0]), np.array([1.0, 10.0]), np.linspace(0, 10.0, 101))  # 10.0 Hz, split up into blocks
exit()


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
