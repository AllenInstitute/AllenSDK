import pytest
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.ecephys_api import EcephysApi
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
def raw_mean_waveforms():
    return {
        0: np.zeros((20, 3)),
        1: np.zeros((20, 3)) + 1,
        2: np.zeros((20, 3)) + 2
    }


@pytest.fixture
def raw_channels():
    return pd.DataFrame({
        'local_index': [0, 1, 2],
        'probe_horizontal_position': [5, 10, 15],
        'probe_id': [0, 0, 0],
        'probe_vertical_position': [10, 22, 10],
        'valid_data': [True, True, False]
    }, index=pd.Index(name='channel_id', data=[0, 1, 2]))


@pytest.fixture
def raw_units():
    return pd.DataFrame({
        'firing_rate': np.linspace(1, 3, 3),
        'isi_violations': [40, 0.5, 0.1],
        'local_index': [0, 0, 1],
        'peak_channel_id': [0, 1, 2],
        'quality': ['good', 'good', 'noise'],
        'snr': [0.1, 1.4, 10.0]
    }, index=pd.Index(name='unit_id', data=np.arange(3)[::-1]))


@pytest.fixture
def raw_probes():
    return pd.DataFrame({
        'description': ['probeA', 'probeB'],
        'location': ['VISp', 'VISam']
    }, index=pd.Index(name='id', data=[0, 1]))


@pytest.fixture
def just_stimulus_table_api(raw_stimulus_table):
    class EcephysJustStimulusTableApi(EcephysApi):
        def get_stimulus_table(self):
            return raw_stimulus_table
    return EcephysJustStimulusTableApi()


@pytest.fixture
def mean_waveforms_api(raw_mean_waveforms, raw_channels, raw_units, raw_probes):
    class EcephysMeanWaveformsApi(EcephysApi):
        def get_mean_waveforms(self):
            return raw_mean_waveforms
        def get_channels(self):
            return raw_channels
        def get_units(self):
            return raw_units
        def get_probes(self):
            return raw_probes
    return EcephysMeanWaveformsApi()


def test_build_stimulus_sweeps(just_stimulus_table_api):
    expected_columns = [
        'start_time', 'stop_time', 'stimulus_name', 'stimulus_block', 'TF', 'SF', 'Ori', 'Contrast', 
        'Pos_x', 'Pos_y', 'Color', 'Image', 'Phase', 'is_movie'
    ]

    session = EcephysSession(api=just_stimulus_table_api)
    obtained = session.stimulus_sweeps

    assert set(expected_columns) == set(obtained.columns)
    assert 'stimulus_sweep_id' == obtained.index.name
    assert 4 == obtained.shape[0]
    assert np.allclose([False, False, False, True], obtained['is_movie'])


def test_build_mean_waveforms(mean_waveforms_api):
    session = EcephysSession(api=mean_waveforms_api)
    obtained = session.mean_waveforms

    assert np.allclose(np.zeros((20, 3)) + 2, obtained['mean_waveforms'].loc[2, :, :])
    assert np.allclose(np.zeros((20, 3)) + 1, obtained['mean_waveforms'].loc[1, :, :])