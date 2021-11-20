import pytest
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysSessionApi


class MockSessionApi(EcephysSessionApi):
    """Mock Data to create an EcephysSession object and pass it into stimulus analysis"""
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