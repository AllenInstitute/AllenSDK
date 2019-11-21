import pytest
import pandas as pd
import numpy as np
import xarray as xr
import types

from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession, nan_intervals, build_spike_histogram


@pytest.fixture
def raw_stimulus_table():
    return pd.DataFrame({
        'start_time': np.arange(4)/2,
        'stop_time':np.arange(1, 5)/2,
        'stimulus_name':['a', 'a', 'a', 'a_movie'],
        'stimulus_block':[0, 0, 0, 1],
        'TF': np.empty(4) * np.nan,
        'SF':np.empty(4) * np.nan,
        'Ori': np.empty(4) * np.nan,
        'Contrast': np.empty(4) * np.nan,
        'Pos_x': np.empty(4) * np.nan,
        'Pos_y': np.empty(4) * np.nan,
        'stimulus_index': [0, 0, 1, 1],
        'Color': np.arange(4)*5.5,
        'Image': np.empty(4) * np.nan,
        'Phase': np.linspace(0, 180, 4),
        "texRes": np.ones([4])
    }, index=pd.Index(name='id', data=np.arange(4)))

@pytest.fixture
def raw_invalid_times_table():
    return pd.DataFrame({
        "start_time": [0.3, 1.1, 1.6],
        "stop_time": [0.6, 1.54, 2.3],
        "tags":
            [
                ["EcephysSession", "739448407", "stimulus"],
                ["EcephysProbe", "123448407", "probeA"],
                ["EcephysProbe", "123448407", "all_probes"],
            ]
    })


@pytest.fixture
def raw_spike_times():
    return {
        0: np.array([5, 6, 7, 8]),
        1: np.array([2.5]),
        2: np.array([1.01, 1.03, 1.02])
    }



@pytest.fixture
def raw_mean_waveforms():
    return {
        0: np.zeros((3, 20)),
        1: np.zeros((3, 20)) + 1,
        2: np.zeros((3, 20)) + 2
    }


@pytest.fixture
def raw_channels():
    return pd.DataFrame({
        'local_index': [0, 1, 2],
        'probe_horizontal_position': [5, 10, 15],
        'probe_id': [0, 0, 0],
        'probe_vertical_position': [10, 22, 33],
        'valid_data': [False, True, True]
    }, index=pd.Index(name='channel_id', data=[0, 1, 2]))


@pytest.fixture
def raw_units():
    return pd.DataFrame({
        'firing_rate': np.linspace(1, 3, 3),
        'isi_violations': [40, 0.5, 0.1],
        'local_index': [0, 0, 1],
        'peak_channel_id': [2, 1, 0],
        'quality': ['good', 'good', 'noise'],
        'snr': [0.1, 1.4, 10.0],
        'on_screen_rf': [True, False, True],
        'p_value_rf': [0.001, 0.01, 0.05]
    }, index=pd.Index(name='unit_id', data=np.arange(3)[::-1]))


@pytest.fixture
def raw_probes():
    return pd.DataFrame({
        'description': ['probeA', 'probeB'],
        'location': ['VISp', 'VISam'],
        'sampling_rate': [30000.0, 30000.0]
    }, index=pd.Index(name='id', data=[0, 1]))


@pytest.fixture
def raw_lfp():
    return {
        0: xr.DataArray(
            data=np.array([[1, 2, 3, 4, 5],
                           [6, 7, 8, 9, 10]]),
            dims=['channel', 'time'],
            coords=[[2, 1], np.linspace(0, 2, 5)]
        )
    }

@pytest.fixture
def just_stimulus_table_api(raw_stimulus_table):
    class EcephysJustStimulusTableApi(EcephysSessionApi):
        def get_stimulus_presentations(self):
            return raw_stimulus_table
        def get_invalid_times(self):
            return pd.DataFrame()
    return EcephysJustStimulusTableApi()


@pytest.fixture
def channels_table_api(raw_channels, raw_probes, raw_lfp, raw_stimulus_table):
    class EcephysChannelsTableApi(EcephysSessionApi):
        def get_channels(self):
            return raw_channels
        def get_probes(self):
            return raw_probes
        def get_lfp(self, pid):
            return raw_lfp[pid]
        def get_stimulus_presentations(self):
            return raw_stimulus_table
        def get_invalid_times(self):
            return pd.DataFrame()

    return EcephysChannelsTableApi()


@pytest.fixture
def lfp_masking_api(raw_channels, raw_probes, raw_lfp, raw_stimulus_table, raw_invalid_times_table):
    class EcephysMaskInvalidLFPApi(EcephysSessionApi):
        def get_channels(self):
            return raw_channels
        def get_probes(self):
            return raw_probes
        def get_lfp(self, pid):
            return raw_lfp[pid]
        def get_stimulus_presentations(self):
            return raw_stimulus_table
        def get_invalid_times(self):
            return raw_invalid_times_table
    return EcephysMaskInvalidLFPApi()


@pytest.fixture
def units_table_api(raw_channels, raw_units, raw_probes):
    class EcephysUnitsTableApi(EcephysSessionApi):
        def get_channels(self):
            return raw_channels
        def get_units(self):
            return raw_units
        def get_probes(self):
            return raw_probes  
    return EcephysUnitsTableApi()

@pytest.fixture
def valid_stimulus_table_api(raw_stimulus_table,raw_invalid_times_table):
    class EcephysValidStimulusTableApi(EcephysSessionApi):
        def get_invalid_times(self):
            return raw_invalid_times_table
        def get_stimulus_presentations(self):
            return raw_stimulus_table
    return EcephysValidStimulusTableApi()


@pytest.fixture
def mean_waveforms_api(raw_mean_waveforms, raw_channels, raw_units, raw_probes):
    class EcephysMeanWaveformsApi(EcephysSessionApi):
        def get_mean_waveforms(self):
            return raw_mean_waveforms
        def get_channels(self):
            return raw_channels
        def get_units(self):
            return raw_units
        def get_probes(self):
            return raw_probes
    return EcephysMeanWaveformsApi()


@pytest.fixture
def spike_times_api(raw_units, raw_channels, raw_probes, raw_stimulus_table, raw_spike_times):
    class EcephysSpikeTimesApi(EcephysSessionApi):
        def get_spike_times(self):
            return raw_spike_times
        def get_channels(self):
            return raw_channels
        def get_units(self):
            return raw_units
        def get_probes(self):
            return raw_probes
        def get_stimulus_presentations(self):
            return raw_stimulus_table

        def get_invalid_times(self):
            return pd.DataFrame()

    return EcephysSpikeTimesApi()


def get_no_spikes_times(self):
    # A special method used for testing cases when there are no spikes for a given session, will be swapped out for
    # get_spike_times()
    return {
        0: np.array([]),
        1: np.array([]),
        2: np.array([])
    }


@pytest.fixture
def session_metadata_api():
    class EcephysSessionMetadataApi(EcephysSessionApi):
        def get_ecephys_session_id(self):
            return 12345
    return EcephysSessionMetadataApi()


def test_get_stimulus_epochs(just_stimulus_table_api):

    expected = pd.DataFrame({
        "start_time": [0, 3/2],
        "stop_time": [3/2, 2],
        "duration": [3/2, 1/2],
        "stimulus_name": ["a", "a_movie"],
        "stimulus_block": [0, 1]
    })

    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_stimulus_epochs()

    print(expected)
    print(obtained)

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_invalid_times(valid_stimulus_table_api, raw_invalid_times_table):

    expected = raw_invalid_times_table

    session = EcephysSession(api=valid_stimulus_table_api)

    obtained = session.get_invalid_times()

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_stimulus_presentations(valid_stimulus_table_api):

    expected = pd.DataFrame({
        "start_time": [0, 1/2, 1, 3/2],
        "stop_time": [1/2, 1, 3/2, 2],
        "stimulus_name": ['invalid_presentation', 'invalid_presentation', 'a', 'a_movie'],
        "phase": [np.nan, np.nan, 120.0, 180.0]
    }, index=pd.Index(name='stimulus_presentations_id', data=[0, 1, 2, 3]))

    session = EcephysSession(api=valid_stimulus_table_api)
    obtained = session.stimulus_presentations[["start_time", "stop_time", "stimulus_name", "phase"]]

    print(expected)
    print(obtained)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_stimulus_presentations_no_invalid_times(just_stimulus_table_api):

    expected = pd.DataFrame({
        "start_time": [0, 1/2, 1, 3/2],
        "stop_time": [1/2, 1, 3/2, 2],
        'stimulus_name': ['a', 'a', 'a', 'a_movie'],

    }, index=pd.Index(name='stimulus_presentations_id', data=[0, 1, 2, 3]))

    session = EcephysSession(api=just_stimulus_table_api)

    obtained = session.stimulus_presentations[["start_time", "stop_time", "stimulus_name"]]
    print(expected)
    print(obtained)

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)

def test_session_metadata(session_metadata_api):
    session = EcephysSession(api=session_metadata_api)

    assert 12345 == session.ecephys_session_id


def test_build_stimulus_presentations(just_stimulus_table_api):
    expected_columns = [
        'start_time', 'stop_time', 'stimulus_name', 'stimulus_block', 
        'temporal_frequency', 'spatial_frequency', 'orientation', 'contrast', 
        'x_position', 'y_position', 'color', 'frame', 'phase', 'duration', "stimulus_condition_id"
    ]

    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.stimulus_presentations

    print(obtained.head())
    print(obtained.columns)

    assert set(expected_columns) == set(obtained.columns)
    assert 'stimulus_presentation_id' == obtained.index.name
    assert 4 == obtained.shape[0]


def test_build_mean_waveforms(mean_waveforms_api):
    session = EcephysSession(api=mean_waveforms_api)
    obtained = session.mean_waveforms

    assert np.allclose(np.zeros((3, 20)) + 2, obtained[2])
    assert np.allclose(np.zeros((3, 20)) + 1, obtained[1])


def test_build_units_table(units_table_api):
    session = EcephysSession(api=units_table_api)
    obtained = session.units

    assert 3 == session.num_units
    assert np.allclose([10, 22, 33], obtained['probe_vertical_position'])
    assert np.allclose([0, 1, 2], obtained.index.values)
    assert np.allclose([0.05, 0.01, 0.001], obtained['p_value_rf'].values)


def test_presentationwise_spike_counts(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.presentationwise_spike_counts(np.linspace(-.1, .1, 3), session.stimulus_presentations.index.values, session.units.index.values)

    first = obtained.loc[{'unit_id': 2, 'stimulus_presentation_id': 2}]
    assert np.allclose([0, 3], first)

    second = obtained.loc[{'unit_id': 1, 'stimulus_presentation_id': 3}]
    assert np.allclose([0, 0], second)

    assert np.allclose([4, 2, 3], obtained.shape)


@pytest.mark.parametrize("spike_times,time_domain,expected", [
    [
        {1: [1.5, 2.5]}, 
        [[1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1]],
        np.array([[1, 1, 0], [1, 1, 0]])[:, :, None]
    ],
    [
        {1: [1.5, 2.5]}, 
        [[1, 2, 3, 4], [1.6, 2.0, 4.0, 4.1]],
        np.array([[1, 1, 0], [0, 1, 0]])[:, :, None]
    ],
    [
        {1: [1.5, 2.5], 2: [1.5, 2.5]}, 
        [[1, 2, 3, 4], [1.6, 2.0, 4.0, 4.1]],
        np.stack(([[1, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 1, 0]]), axis=2)
    ]
,
    [
        {1: [1.5, 2.5], 2: [1.5, 1.55]}, 
        [[1, 2, 3, 4], [1.6, 2.0, 4.0, 4.1]],
        np.stack(([[1, 1, 0], [0, 1, 0]], [[2, 0, 0], [0, 0, 0]]), axis=2)
    ]
])
@pytest.mark.parametrize("binarize", [True, False])
def test_build_spike_histogram(spike_times, time_domain, expected, binarize):
    
    unit_ids = [k for k in spike_times.keys()]
    obtained = build_spike_histogram(time_domain, spike_times, unit_ids, binarize=binarize)

    expected = np.array(expected)
    if binarize:
        expected[expected > 0] = 1

    print(expected - obtained)
    assert np.allclose(expected, obtained)


def test_presentationwise_spike_times(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.presentationwise_spike_times(session.stimulus_presentations.index.values, session.units.index.values)

    expected = pd.DataFrame({
        'unit_id': [2, 2, 2],
        'stimulus_presentation_id': [2, 2, 2, ],
        'time_since_stimulus_presentation_onset': [0.01, 0.02, 0.03]
    }, index=pd.Index(name='spike_time', data=[1.01, 1.02, 1.03]))

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)    


def test_empty_presentationwise_spike_times(spike_times_api):
    # Test that when there are no spikes presentationwise_spike_times doesn't fail and instead returns a empty dataframe
    spike_times_api.get_spike_times = types.MethodType(get_no_spikes_times, spike_times_api)
    session = EcephysSession(api=spike_times_api)
    obtained = session.presentationwise_spike_times(session.stimulus_presentations.index.values,
                                                    session.units.index.values)
    assert(isinstance(obtained, pd.DataFrame))
    assert(obtained.empty)


def test_conditionwise_spike_statistics(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.conditionwise_spike_statistics(stimulus_presentation_ids=[0, 1, 2])

    pd.set_option('display.max_columns', None)

    assert obtained.loc[(2, 2), "spike_count"] == 3
    assert obtained.loc[(2, 2), "stimulus_presentation_count"] == 1


def test_conditionwise_spike_statistics_using_rates(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.conditionwise_spike_statistics(stimulus_presentation_ids=[0, 1, 2], use_rates=True)

    pd.set_option('display.max_columns', None)
    assert np.allclose([0, 0, 6], obtained["spike_mean"].values)


def test_empty_conditionwise_spike_statistics(spike_times_api):
    # special case when there are no spikes
    spike_times_api.get_spike_times = types.MethodType(get_no_spikes_times, spike_times_api)
    session = EcephysSession(api=spike_times_api)
    obtained = session.conditionwise_spike_statistics(
        stimulus_presentation_ids=session.stimulus_presentations.index.values,
        unit_ids=session.units.index.values
    )
    assert(len(obtained) == 12)
    assert(not np.any(obtained['spike_count']))  # check all spike_counts are 0
    assert(not np.any(obtained['spike_mean']))  # spike_means are 0
    assert(np.all(np.isnan(obtained['spike_std'])))  # std/sem will be undefined
    assert(np.all(np.isnan(obtained['spike_sem'])))


def test_get_stimulus_parameter_values(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_stimulus_parameter_values()

    expected = {
        'color': [0, 5.5, 11, 16.5],
        'phase': [0, 60, 120, 180]
    }
    
    for k, v in expected.items():
        assert np.allclose(v, obtained[k])
    assert len(expected) == len(obtained)


@pytest.mark.parametrize("detailed", [True, False])
def test_get_stimulus_table(detailed, just_stimulus_table_api, raw_stimulus_table):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_stimulus_table(['a'], include_detailed_parameters=detailed)

    expected_columns = ['start_time', 'stop_time', 'stimulus_name', 'stimulus_block', 'Color', 'Phase']
    if detailed:
        expected_columns.append("texRes")
    expected = raw_stimulus_table.loc[:2, expected_columns]

    expected['duration'] = expected['stop_time'] - expected['start_time']
    expected["stimulus_condition_id"] = [0, 1, 2]
    expected.rename(columns={"Color": "color", "Phase": "phase"}, inplace=True)

    print(expected)
    print(obtained)

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_filter_owned_df(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    ids = [0, 2]
    obtained = session._filter_owned_df('stimulus_presentations', ids)

    assert np.allclose([0, 120], obtained['phase'].values)


def test_filter_owned_df_scalar(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    ids = 3

    obtained = session._filter_owned_df('stimulus_presentations', ids)
    assert obtained['phase'].values[0] == 180


def test_build_inter_presentation_intervals(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.inter_presentation_intervals

    expected = pd.DataFrame({
            'interval': [0, 0, 0]
        }, index=pd.MultiIndex(
            levels=[[0, 1, 2], [1, 2, 3]],
            codes=[[0, 1, 2], [0, 1, 2]],
            names=['from_presentation_id', 'to_presentation_id']
        )
    )

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_inter_presentation_intervals_for_stimulus(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_inter_presentation_intervals_for_stimulus('a')

    expected = pd.DataFrame({
            'interval': [0, 0]
        }, index=pd.MultiIndex(
            levels=[[0, 1], [1, 2]],
            codes=[[0, 1], [0, 1]],
            names=['from_presentation_id', 'to_presentation_id']
        )
    )

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_lfp(channels_table_api):
    session = EcephysSession(api=channels_table_api)
    obtained = session.get_lfp(0)

    expected = xr.DataArray(
        data=np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10]]),
        dims=['channel', 'time'],
        coords=[[2, 1], np.linspace(0, 2, 5)]
    )

    xr.testing.assert_equal(expected, obtained)


def test_get_lfp_mask_invalid(lfp_masking_api):
    session = EcephysSession(api=lfp_masking_api)
    obtained = session.get_lfp(0)

    expected = xr.DataArray(
        data=np.array([[1, 2, 3, np.nan, np.nan],
                       [6, 7, 8, np.nan, np.nan]]),
        dims=['channel', 'time'],
        coords=[[2, 1], np.linspace(0, 2, 5)]
    )
    print(expected)
    print(obtained)

    xr.testing.assert_equal(expected, obtained)


@pytest.mark.parametrize("inp,expected", [
    [[np.nan, np.nan, 4, 4, 4, 5, 5], [0, 2, 5, 7]]
])
def test_nan_intervals(inp, expected):
    assert np.allclose(
        expected, nan_intervals(inp)
    )
