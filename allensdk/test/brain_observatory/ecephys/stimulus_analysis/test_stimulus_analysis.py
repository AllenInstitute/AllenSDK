import pytest
import pandas as pd
import numpy as np
import xarray as xr
import warnings

from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.stimulus_analysis import StimulusAnalysis, \
    running_modulation, lifetime_sparseness, fano_factor, overall_firing_rate, get_fr, osi, dsi


pd.set_option('display.max_columns', None)


class MockSessionApi(EcephysSessionApi):
    """Mock Data to create an EcephysSession object and pass it into stimulus analysis

    # TODO: move to conftest so other tests can use data
    """
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
            'firing_rate': np.linspace(1, 3, 6),
            'isi_violations': [40, 0.5, 0.1, 0.2, 0.0, 0.1],
            'local_index': [0, 0, 1, 1, 2, 2],
            'peak_channel_id': [0, 2, 1, 1, 2, 0],
            'quality': ['good', 'good', 'good', 'bad', 'good', 'good'],
        }, index=pd.Index(name='unit_id', data=np.arange(6)[::-1]))
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
            'stimulus_name': ['spontaneous'] + ['s0'] * 6 + ['spontaneous'] + ['s1'] * 2,
            'stimulus_block': [0] + [1] * 6 + [0] + [2] * 2,
            'duration': 0.5,
            'stimulus_index': [0] + [1] * 6 + [0] + [2] * 2,
            'conditions': [0, 0, 0, 0, 1, 1, 1, 0, 2, 3]  # generic stimulus condition
        }, index=pd.Index(name='id', data=np.arange(10)))

    def get_invalid_times(self):
        return pd.DataFrame()

    def get_running_speed(self):
        return pd.DataFrame({
            "start_time": np.linspace(0.0, 9.9, 100),
            "end_time": np.linspace(0.1, 10.0, 100),
            "velocity": np.linspace(-0.1, 11.0, 100)
        })


@pytest.fixture
def ecephys_api():
    return MockSessionApi()


def test_unit_ids(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session)
    assert(set(stim_analysis.unit_ids) == set(range(6)))
    assert(stim_analysis.unit_count == 6)


def test_unit_ids_filter_by_id(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, filter=[2, 3, 1])
    assert(set(stim_analysis.unit_ids) == {1, 2, 3})
    assert(stim_analysis.unit_count == 3)

    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'unit_id': [3, 0]})
    assert(set(stim_analysis.unit_ids) == {0, 3})
    assert(stim_analysis.unit_count == 2)

    with pytest.raises(KeyError):
        # If unit ids don't exists should raise an error
        stim_analysis = StimulusAnalysis(ecephys_session=session, filter=[100, 200])
        units = stim_analysis.unit_ids


def test_unit_ids_filtered(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'VISp'})
    assert(set(stim_analysis.unit_ids) == {0, 2, 3, 5})
    assert(stim_analysis.unit_count == 4)

    stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'VISp', 'quality': 'good'})
    assert(set(stim_analysis.unit_ids) == {0, 3, 5})
    assert(stim_analysis.unit_count == 3)

    with pytest.raises(Exception):
        # No units found should raise exception
        stim_analysis = StimulusAnalysis(ecephys_session=session, filter={'location': 'pSIV'})
        stim_analysis.unit_ids
        stim_analysis.unit_count


def test_stim_table(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    assert(isinstance(stim_analysis.stim_table, pd.DataFrame))
    assert(len(stim_analysis.stim_table) == 6)
    assert(stim_analysis.total_presentations == 6)

    # Make sure certain columns exist
    assert('start_time' in stim_analysis.stim_table)
    assert('stop_time' in stim_analysis.stim_table)
    assert('stimulus_condition_id' in stim_analysis.stim_table)
    assert('stimulus_name' in stim_analysis.stim_table)
    assert('duration' in stim_analysis.stim_table)

    with pytest.raises(Exception):
        stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='0s')
        stim_analysis.stim_table


def test_stim_table_spontaneous(ecephys_api):
    # By default table should be empty because non of the stimulus are above the duration threshold
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, spontaneous_threshold=0.49)
    assert(isinstance(stim_analysis.stim_table_spontaneous, pd.DataFrame))
    assert(len(stim_analysis.stim_table_spontaneous) == 2)

    # Check that threshold is working
    stim_analysis = StimulusAnalysis(ecephys_session=session, spontaneous_threshold=0.51)
    assert(len(stim_analysis.stim_table_spontaneous) == 0)


def test_conditionwise_psth(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5,
                                     psth_resolution=0.1)
    assert(isinstance(stim_analysis.conditionwise_psth, xr.DataArray))
    # assert(stim_analysis.conditionwise_psth.shape == (2, 4, 6))
    assert(stim_analysis.conditionwise_psth.coords['time_relative_to_stimulus_onset'].size == 4)  # 0.5/0.1 - 1
    assert(stim_analysis.conditionwise_psth.coords['unit_id'].size == 6)
    assert(stim_analysis.conditionwise_psth.coords['stimulus_condition_id'].size == 2)
    assert(np.allclose(stim_analysis.conditionwise_psth[{'unit_id': 0, 'stimulus_condition_id': 1}].values,
                       np.array([1.0/3.0, 0.0, 0.0, 0.0])))

    # Make sure psth doesn't fail even when all the condition_ids are unique.
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s1', trial_duration=0.5,
                                     psth_resolution=0.1)
    assert(stim_analysis.conditionwise_psth.coords['time_relative_to_stimulus_onset'].size == 4)
    assert(stim_analysis.conditionwise_psth.coords['unit_id'].size == 6)
    assert(stim_analysis.conditionwise_psth.coords['stimulus_condition_id'].size == 2)


def test_conditionwise_statistics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    assert(len(stim_analysis.conditionwise_statistics) == 2*6) # units x condition_ids
    assert(set(stim_analysis.conditionwise_statistics.index.names) == {'unit_id', 'stimulus_condition_id'})
    assert(set(stim_analysis.conditionwise_statistics.columns) ==
           {'spike_std', 'spike_sem', 'spike_count', 'stimulus_presentation_count', 'spike_mean'})

    expected = pd.Series(
        [2.0, 3.0, 0.66666667, 0.57735027, 0.33333333], 
        ["spike_count", "stimulus_presentation_count", "spike_mean", "spike_std", "spike_sem"]
    )
    obtained = stim_analysis.conditionwise_statistics.loc[(0, 1)]
    pd.testing.assert_series_equal(expected, obtained[expected.index], check_less_precise=5, check_names=False)


def test_presentationwise_spike_times(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    assert(len(stim_analysis.presentationwise_spike_times) == 12)
    assert(list(stim_analysis.presentationwise_spike_times.index.names) == ['spike_time'])
    assert(set(stim_analysis.presentationwise_spike_times.columns) == {'stimulus_presentation_id', 'unit_id', 'time_since_stimulus_presentation_onset'})
    assert(stim_analysis.presentationwise_spike_times.loc[1.01]['unit_id'] == 2)
    assert(stim_analysis.presentationwise_spike_times.loc[1.01]['stimulus_presentation_id'] == 2)
    assert(len(stim_analysis.presentationwise_spike_times.loc[3.0]) == 2)


def test_presentationwise_statistics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5)
    assert(len(stim_analysis.presentationwise_statistics) == 6*6)  # units x presentation_ids
    assert(set(stim_analysis.presentationwise_statistics.index.names) == {'stimulus_presentation_id', 'unit_id'})
    assert(set(stim_analysis.presentationwise_statistics.columns) == {'spike_counts', 'stimulus_condition_id',
                                                                      'running_speed'})
    assert(stim_analysis.presentationwise_statistics.loc[1, 0]['spike_counts'] == 1.0)
    assert(stim_analysis.presentationwise_statistics.loc[1, 0]['stimulus_condition_id'] == 1.0)
    assert(np.isclose(stim_analysis.presentationwise_statistics.loc[1, 0]['running_speed'], 0.684848))


def test_stimulus_conditions(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5)
    assert(len(stim_analysis.stimulus_conditions) == 2)
    assert(np.all(stim_analysis.stimulus_conditions['stimulus_name'].unique() == ['s0']))
    assert(set(stim_analysis.stimulus_conditions['conditions'].unique()) == {0, 1})


def test_running_speed(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    assert(set(stim_analysis.running_speed.index.values) == set(range(1, 7)))
    assert(np.isclose(stim_analysis.running_speed.loc[1]['running_speed'], 0.684848))
    assert(np.isclose(stim_analysis.running_speed.loc[3]['running_speed'], 1.806061))
    assert(np.isclose(stim_analysis.running_speed.loc[6]['running_speed'], 3.487879))


def test_spikes(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    assert(isinstance(stim_analysis.spikes, dict))
    assert(stim_analysis.spikes.keys() == set(range(6)))
    assert(np.allclose(stim_analysis.spikes[0], [1, 2, 3, 4]))
    assert(np.allclose(stim_analysis.spikes[4], [0.01, 1.7 , 2.13, 3.19, 4.25]))
    assert(stim_analysis.spikes[3].size == 0)

    # Check that spikes dict is filtering units
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', filter=[0, 2])
    assert(stim_analysis.spikes.keys() == {0, 2})


def test_get_preferred_condition(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')
    assert(stim_analysis._get_preferred_condition(3) == 1)

    with pytest.raises(KeyError):
        stim_analysis._get_preferred_condition(10)

def test_check_multiple_preferred_conditions(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0')

    assert(stim_analysis._check_multiple_pref_conditions(0, 'conditions', [0, 1]) is False)
    assert(stim_analysis._check_multiple_pref_conditions(3, 'conditions', [0, 1]) is True)


def test_get_time_to_peak(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    stim_analysis = StimulusAnalysis(ecephys_session=session, stimulus_key='s0', trial_duration=0.5)
    assert(stim_analysis._get_time_to_peak(1, stim_analysis._get_preferred_condition(1)) == 0.0005)


@pytest.mark.parametrize('spike_counts,running_speeds,speed_threshold,expected',
                         [
                             (np.zeros(10), np.zeros(1),1.0, (np.nan, np.nan)),  # Input error, return nan
                             (np.zeros(5), np.full(5, 2.0), 1.0, (np.nan, np.nan)),  # returns Nan, always running
                             (np.zeros(5), np.full(5, 2.0), 2.1, (np.nan, np.nan)),  # returns Nan, always stationary
                             (np.zeros(5), np.array([0.0, 0.0, 2.0, 2.5, 3.0]), 1.0, (np.nan, np.nan)), # No firing, return Nans
                             (np.ones(5), np.array([0.0, 0.0, 2.0, 2.0, 2.0]), 1.0, (np.nan, 0.0)),  # always the same fr, pval is Nan but run_mod is 0.0)
                             (np.array([3.0, 3.0, 1.5, 1.5, 0.9]), np.array([0.0, 0.0, 2.0, 2.5, 3.0]), 1.0, (0.013559949584378913, -0.5666666666666667)),
                             (np.array([3.0, 3.0, 1.5, 1.5, 1.5]), np.array([0.0, 0.0, 2.0, 2.5, 3.0]), 1.0, (0.0, -0.5)),
                             (np.array([3.0, 3.0, 1.5, 5.5, 2.5]), np.array([0.0, 0.0, 2.0, 2.5, 3.0]), 1.0, (0.9024099927051468, 0.052631578947368376)),
                             (np.array([0.0, 0.0, 4.0, 4.0]), np.array([0.0, 0.0, 2.5, 3.0]), 1.0, (0.0, 1.0))
                         ])
def test_running_modulation(spike_counts, running_speeds, speed_threshold, expected):
    rm = running_modulation(spike_counts, running_speeds, speed_threshold)
    assert(np.allclose(rm, expected, equal_nan=True))


@pytest.mark.parametrize('responses,expected',
                         [
                             (np.array([1.0]), np.nan),  # can't calculate for single point
                             (np.full(20, 3.2), 0.0),  # always the same, sparness should be at/near 0
                             (np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.0),  # spareness should be close to 1
                             (np.array([2.24, 3.6, 0.8, 2.4, 3.52, 5.68, 8.96, 0.0]), 0.43500091849856115)
                         ])
def test_lifetime_sparseness(responses, expected):
    lts = lifetime_sparseness(responses)
    assert(np.isclose(lts, expected, equal_nan=True))


@pytest.mark.parametrize('spike_counts,expected',
                         [
                             (np.zeros(10), np.nan),  # mean 0.0 leads to Nan
                             (np.array([-1.5, 1.5]), np.nan),  # mean 0
                             (np.ones(5), 0.0),  # no variance
                             (np.array([1.2, 20.0, 0.0, 36.2, 0.6]), 17.921379310344832),  # High variance
                             (np.array([5.1, 5.3, 5.2, 5.1, 5.2]), 0.0010810810810810846), # low variance
                         ])
def test_fano_factor(spike_counts, expected):
    ff = fano_factor(spike_counts)
    assert(np.isclose(ff, expected, equal_nan=True))


@pytest.mark.parametrize('start_times,stop_times,spike_times,expected',
                         [
                             (np.array([0.0]), np.array([0.0]), np.linspace(0, 10.0, 10), np.nan), # nan, total time 0.0
                             (np.arange(1.0, 3.0), np.arange(0.0, 2.0), np.linspace(0, 10.0, 10), np.nan),  # nan, total_time negative
                             (np.arange(1.0, 4.0), np.arange(0.0, 2.0), np.linspace(0, 10.0, 10), np.nan),  # nan, time lengths don't match
                             (np.array([0.0]), np.array([1.0]), np.linspace(0, 10.0, 100), 10.0),
                             (np.array([0.0, 9.0]), np.array([1.0, 10.0]), np.linspace(0, 10.0, 101), 10.0)  # 10.0 Hz split up into blocks
                         ])
def test_overall_firing_rate(start_times, stop_times, spike_times, expected):
    ofr = overall_firing_rate(start_times, stop_times, spike_times)
    assert(np.isclose(ofr, expected, equal_nan=True))


@pytest.mark.parametrize('spikes,sampling_freq,sweep_length,expected',
                         [
                             (np.array([0.82764702, 0.83624702, 1.09211374]), 10, 1.5, [0.0, 0.0, 0.0, 0.0, 0.000133830, 0.004431861, 0.05412495, 0.2464033072, 0.45293459677, 0.4839428913,
                                                                                        0.452934596, 0.2464033072, 0.054124958, 0.00443186162, 0.0001338306]),
                             (np.array([]), 10, 1.5, np.zeros(15))
                         ])
def test_get_fr(spikes, sampling_freq, sweep_length, expected):
    frs = get_fr(spikes, num_timestep_second=sampling_freq, sweep_length=sweep_length)
    assert(len(frs) == int(sampling_freq*sweep_length))
    assert(np.allclose(frs, expected))


@pytest.mark.parametrize('orivals,tuning,expected',
                         [
                             (np.array([1.0]), np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]), np.nan),
                             (np.array([]), np.array([]), np.nan),
                             (np.array([0.0+0.0j, 0.52359878 + 0.j, 1.04719755 + 0.j, 1.57079633+0.j, 2.0943951+0.j, 2.61799388+0.j]), np.array([5.5, 4.44, 3.54166667, 4.10869565, 4.42, 4.55319149]), 0.07873455094232604)
                         ])
def test_osi(orivals, tuning, expected):
    osi_val = osi(orivals, tuning)
    assert(np.allclose(osi_val, expected, equal_nan=True))


@pytest.mark.parametrize('orivals,tuning,expected',
                         [
                             (np.array([1.0]), np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]), np.nan),
                             (np.array([]), np.array([]), np.nan),
                             (np.array([0.0+0.0j, 0.52359878 + 0.j, 1.04719755 + 0.j, 1.57079633+0.j, 2.0943951+0.j, 2.61799388+0.j]), np.array([5.5, 4.44, 3.54166667, 4.10869565, 4.42, 4.55319149]), 0.6126966469601506)
                         ])
def test_dsi(orivals, tuning, expected):
    dsi_val = dsi(orivals, tuning)
    assert(np.allclose(dsi_val, expected, equal_nan=True))


if __name__ == '__main__':
    # test_unit_ids()
    # test_unit_ids_filter_by_id()
    # test_unit_ids_filtered()
    # test_stim_table()
    # test_stim_table_spontaneous()
    # test_conditionwise_psth()
    # test_conditionwise_statistics()
    # test_presentationwise_spike_times()
    # test_presentationwise_statistics()
    # test_stimulus_conditions()
    # test_running_speed()
    # test_spikes()
    # test_get_preferred_condition()
    # test_get_time_to_peak()
    # test_running_modulation(spike_counts=np.zeros(10), running_speeds=np.zeros(1), speed_threshold=1.0,
    #                         expected=(np.nan, np.nan))
    # test_lifetime_sparseness(np.array([1.0]), 1.0)
    # test_fano_factor([-1.5, 1.5], np.nan)  # mean 0.0 leads to Nan
    # test_overall_firing_rate(np.array([0.0]), np.array([0.0]), np.linspace(0, 10.0, 10))
    # test_osi(np.array([1.0]), np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0]), np.nan)
    test_osi(np.array([0.0+0.0j, 0.52359878 + 0.j, 1.04719755 + 0.j, 1.57079633+0.j, 2.0943951+0.j, 2.61799388+0.j]),
             np.array([5.5, 4.44, 3.54166667, 4.10869565, 4.42, 4.55319149]), 0.07873455094232604)
    pass