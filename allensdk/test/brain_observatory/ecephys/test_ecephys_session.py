import pytest
import pandas as pd
import numpy as np
import xarray as xr

from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


@pytest.fixture
def raw_stimulus_table():
    return pd.DataFrame({
        'start_time': np.arange(4)/2,
        'stop_time':np.arange(1, 5)/2,
        'stimulus_name':['a', 'a', 'a', 'a_movie'],
        'stimulus_block':[0, 0, 1, 1],
        'TF': np.empty(4) * np.nan,
        'SF':np.empty(4) * np.nan,
        'Ori': np.empty(4) * np.nan,
        'Contrast': np.empty(4) * np.nan,
        'Pos_x': np.empty(4) * np.nan,
        'Pos_y': np.empty(4) * np.nan,
        'stimulus_index': [0, 0, 1, 1],
        'Color': np.arange(4)*5.5,
        'Image': np.empty(4) * np.nan,
        'Phase': np.linspace(0, 180, 4)
    }, index=pd.Index(name='id', data=np.arange(4)))


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
        'snr': [0.1, 1.4, 10.0]
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
            data=np.array([[1, 2, 3], [4, 5, 6]]),
            dims=['channel', 'time'],
            coords=[[2, 1], np.linspace(0, 1, 3)]
        )
    }

@pytest.fixture
def just_stimulus_table_api(raw_stimulus_table):
    class EcephysJustStimulusTableApi(EcephysSessionApi):
        def get_stimulus_presentations(self):
            return raw_stimulus_table
    return EcephysJustStimulusTableApi()


@pytest.fixture
def channels_table_api(raw_channels, raw_probes, raw_lfp):
    class EcephysChannelsTableApi(EcephysApi):
        def get_channels(self):
            return raw_channels
        def get_probes(self):
            return raw_probes
        def get_lfp(self, pid):
            return raw_lfp[pid]
    return EcephysChannelsTableApi()


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
    return EcephysSpikeTimesApi()


@pytest.fixture
def session_metadata_api():
    class EcephysSessionMetadataApi(EcephysSessionApi):
        def get_ecephys_session_id(self):
            return 12345
    return EcephysSessionMetadataApi()


def test_session_metadata(session_metadata_api):
    session = EcephysSession(api=session_metadata_api)

    assert 12345 == session.ecephys_session_id


def test_build_stimulus_presentations(just_stimulus_table_api):
    expected_columns = [
        'start_time', 'stop_time', 'stimulus_name', 'stimulus_block', 'TF', 'SF', 'Ori', 'Contrast', 
        'Pos_x', 'Pos_y', 'Color', 'Image', 'Phase', 'duration'
    ]

    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.stimulus_presentations

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

    assert 2 == session.num_units
    assert np.allclose([22, 33], obtained['probe_vertical_position'])
    assert np.allclose([1, 2], obtained.index.values)


def test_framewise_spike_counts(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.presentationwise_spike_counts(np.linspace(-.1, .1, 3), session.stimulus_presentations.index.values, session.units.index.values)

    first = obtained['spike_counts'].loc[{'unit_id': 2, 'stimulus_presentation_id': 2}]
    assert np.allclose([0, 3], first)

    second = obtained['spike_counts'].loc[{'unit_id': 1, 'stimulus_presentation_id': 3}]
    assert np.allclose([0, 0], second)

    assert np.allclose([4, 2, 2], obtained['spike_counts'].shape)


def test_presentationwise_spike_times(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.presentationwise_spike_times(session.stimulus_presentations.index.values, session.units.index.values)

    expected = pd.DataFrame({
        'unit_id': [2, 2, 2],
        'stimulus_presentation_id': [2, 2, 2, ]
    }, index=pd.Index(name='spike_time', data=[1.01, 1.02, 1.03]))

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_conditionwise_spike_counts(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.conditionwise_spike_counts(session.stimulus_presentations.index.values, session.units.index.values)

    expected = pd.DataFrame({
        'unit_id': [2],
        'stimulus_name': ['a'],
        'TF': ['null'],
        'SF': ['null'],
        'Ori': ['null'],
        'Contrast': ['null'],
        'Pos_x': ['null'],
        'Pos_y': ['null'],
        'Color': [11],
        'Image': ['null'],
        'Phase': 120,
        'count': 3
    })

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_conditionwise_mean_spike_counts(spike_times_api):
    session = EcephysSession(api=spike_times_api)
    obtained = session.conditionwise_mean_spike_counts(session.stimulus_presentations.index.values, session.units.index.values)

    expected = pd.DataFrame({
        'unit_id': [2],
        'stimulus_name': ['a'],
        'TF': ['null'],
        'SF': ['null'],
        'Ori': ['null'],
        'Contrast': ['null'],
        'Pos_x': ['null'],
        'Pos_y': ['null'],
        'Color': [11],
        'Image': ['null'],
        'Phase': 120,
        'mean_spike_count': 3
    })

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_stimulus_conditions(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_stimulus_conditions(stimulus_presentation_ids=[3])

    expected = pd.DataFrame({
        'Color': [16.5],
        'stimulus_name': ['a_movie'],
        'Phase': [180]
    })

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_get_stimulus_parameter_values(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_stimulus_parameter_values()

    expected = {
        'Color': [0, 5.5, 11, 16.5],
        'Phase': [0, 60, 120, 180]
    }
    
    for k, v in expected.items():
        assert np.allclose(v, obtained[k])
    assert len(expected) == len(obtained)


def test_get_presentations_for_stimulus(just_stimulus_table_api, raw_stimulus_table):
    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.get_presentations_for_stimulus(['a'])

    expected = raw_stimulus_table.loc[:2, [
        'start_time', 'stop_time', 'stimulus_name', 'stimulus_block', 'Color', 'Phase'
    ]]
    expected['duration'] = expected['stop_time'] - expected['start_time']

    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_dtype=False)


def test_filter_owned_df(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    ids = [0, 2]
    obtained = session._filter_owned_df('stimulus_presentations', ids)

    assert np.allclose([0, 120], obtained['Phase'].values)


def test_filter_owned_df_scalar(just_stimulus_table_api):
    session = EcephysSession(api=just_stimulus_table_api)
    ids = 3

    with pytest.warns(UserWarning) as w:
        obtained = session._filter_owned_df('stimulus_presentations', ids)

    assert w[-1].message.args[0] == 'a scalar (3) was provided as ids, filtering to a single row of stimulus_presentations.'
    assert obtained['Phase'].values[0] == 180


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
        data=np.array([[1, 2, 3], [4, 5, 6]]),
        dims=['channel', 'time'],
        coords=[[2, 1], np.linspace(0, 1, 3)]
    )

    xr.testing.assert_equal(expected, obtained)