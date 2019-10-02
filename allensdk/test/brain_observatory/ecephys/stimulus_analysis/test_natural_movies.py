import pytest
import numpy as np
import pandas as pd

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.stimulus_analysis.natural_movies import NaturalMovies
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


class MockNMSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        raise NotImplementedError()


@pytest.fixture
def ecephys_api():
    return MockNMSessionApi()


@pytest.mark.skip(reason='NaturalMovies not fully implemented.')
def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    nm = NaturalMovies(ecephys_session=session)
    assert(nm.name == 'Natural Movies')
    assert(set(nm.unit_ids) == set(range(6)))
    # assert(len(nm.conditionwise_statistics) == 119*6)
    # assert(nm.conditionwise_psth.shape == (119, 249, 6))
    # assert(not nm.presentationwise_spike_times.empty)
    # assert(len(nm.presentationwise_statistics) == 119*6)
    # assert(len(nm.stimulus_conditions) == 119)


@pytest.mark.skip(reason='NaturalMovies not fully implemented.')
def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    nm = NaturalMovies(ecephys_session=session)
    assert(isinstance(nm.stim_table, pd.DataFrame))
    # assert(len(nm.stim_table) == 119)
    # assert(set(nm.stim_table.columns).issuperset({'frame', 'start_time', 'stop_time'}))
    # assert(np.all(nm.images == np.arange(-1.0, 118)))
    # assert(nm.number_images == 119)
    # assert(nm.number_nonblank == 118)


@pytest.mark.skip(reason='NaturalMovies not fully implemented.')
def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    nm = NaturalMovies(ecephys_session=session)
    assert(isinstance(nm.metrics, pd.DataFrame))
    assert(len(nm.metrics) == 6)
    assert(nm.metrics.index.names == ['unit_id'])

    assert('fano_nm' in nm.metrics.columns)
    assert('firing_rate_nm' in nm.metrics.columns)
    assert('lifetime_sparseness_nm' in nm.metrics.columns)
    assert('run_pval_nm' in nm.metrics.columns)
    assert('run_mod_nm' in nm.metrics.columns)
