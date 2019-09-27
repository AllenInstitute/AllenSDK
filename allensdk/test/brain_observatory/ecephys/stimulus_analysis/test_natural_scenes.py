import pytest
import numpy as np
import pandas as pd

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.natural_scenes import NaturalScenes, image_selectivity


class MockNSSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        return pd.DataFrame({
            'start_time': np.concatenate(([0.0], np.linspace(0.5, 29.50, 119, endpoint=True), [39.75])),
            'stop_time': np.concatenate(([0.5], np.linspace(0.75, 39.75, 119, endpoint=True), [40.25])),
            'stimulus_name': ['spontaneous'] + ['natural_scenes']*119 + ['spontaneous'],
            'stimulus_block': [0] + [1]*119 + [0],
            'duration': [0.5] + [0.25]*119 + [0.5],
            'stimulus_index': [0] + [1]*119 + [0],
            'frame': np.concatenate(([np.nan], np.arange(-1.0, 118.0), [np.nan]))
        }, index=pd.Index(name='id', data=np.arange(121)))

    def get_invalid_times(self):
        return pd.DataFrame()

@pytest.fixture
def ecephys_api():
    return MockNSSessionApi()


def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    ns = NaturalScenes(ecephys_session=session)
    assert(ns.name == 'Natural Scenes')
    assert(set(ns.unit_ids) == set(range(6)))
    assert(len(ns.conditionwise_statistics) == 119*6)
    assert(ns.conditionwise_psth.shape == (119, 249, 6))
    assert(not ns.presentationwise_spike_times.empty)
    assert(len(ns.presentationwise_statistics) == 119*6)
    assert(len(ns.stimulus_conditions) == 119)


def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    ns = NaturalScenes(ecephys_session=session)
    assert(isinstance(ns.stim_table, pd.DataFrame))
    assert(len(ns.stim_table) == 119)
    assert(set(ns.stim_table.columns).issuperset({'frame', 'start_time', 'stop_time'}))

    assert(np.all(ns.images == np.arange(-1.0, 118)))
    assert(ns.number_images == 119)
    assert(ns.number_nonblank == 118)


def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    ns = NaturalScenes(ecephys_session=session)
    assert(isinstance(ns.metrics, pd.DataFrame))
    assert(len(ns.metrics) == 6)
    assert(ns.metrics.index.names == ['unit_id'])

    assert('pref_image_ns' in ns.metrics.columns)
    assert(np.all(ns.metrics['pref_image_ns'].loc[[0, 1, 3, 4]] == [2, 9, 2, 4]))

    assert('image_selectivity_ns' in ns.metrics.columns)
    assert('firing_rate_ns' in ns.metrics.columns)
    assert('fano_ns' in ns.metrics.columns)
    assert('time_to_peak_ns' in ns.metrics.columns)
    assert('lifetime_sparseness_ns' in ns.metrics.columns)
    assert('run_pval_ns' in ns.metrics.columns)
    assert('run_mod_ns' in ns.metrics.columns)


@pytest.mark.parametrize('responses,expected',
                         [
                             (np.array([]), np.nan),  # invalid input
                             (np.array([1.0]), np.nan),  # selectivity of one image is undefined
                             (np.array([0.0]), np.nan),
                             (np.zeros(118), 0.0),  # responds uniformly
                             (np.ones(118), 0.0),  # responds uniformly
                             (np.array([0.0]*200 + [1.0]), 0.99004975),  # reponse to 1 image ~ 1.0
                             (np.array([5.5, 0.0, 15.0, 10.0, 2.3, 4.9]), 0.16166666666666674)
                         ])
def test_image_selectivity(responses, expected):
    img_sel = image_selectivity(responses)
    assert(np.isclose(img_sel, expected, equal_nan=True))


if __name__ == '__main__':
    test_load()
    # test_stimulus()
    # test_metrics()
