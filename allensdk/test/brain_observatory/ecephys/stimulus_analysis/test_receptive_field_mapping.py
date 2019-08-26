import pytest
import pandas as pd
import numpy as np

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


class MockRFMSessionApi(MockSessionApi):
    def get_stimulus_presentations(self):
        features = np.array(np.meshgrid([30.0, -20.0, 40.0, 20.0, 0.0, -30.0, -40.0, 10.0, -10.0],  # x_position
                                        [10.0, -10.0, 30.0, 40.0, -40.0, -30.0, -20.0, 20.0, 0.0])  # y_position
                            ).reshape(2, 81)

        return pd.DataFrame({
            'start_time': np.concatenate(([0.0], np.linspace(0.5, 20.50, 81, endpoint=True), [20.75])),
            'stop_time': np.concatenate(([0.5], np.linspace(0.75, 20.75, 81, endpoint=True), [21.25])),
            'stimulus_name': ['spontaneous'] + ['gabors']*81 + ['spontaneous'],
            'stimulus_block': [0] + [1]*81 + [0],
            'duration': [0.5] + [0.25]*81 + [0.5],
            'stimulus_index': [0] + [1]*81 + [0],
            'x_position': np.concatenate(([np.nan], features[0, :], [np.nan])),
            'y_position': np.concatenate(([np.nan], features[1, :], [np.nan]))
        }, index=pd.Index(name='id', data=np.arange(83)))


@pytest.fixture
def ecephys_api():
    return MockRFMSessionApi()


def mock_ecephys_api():
    return MockRFMSessionApi()


def test_load(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    rfm = ReceptiveFieldMapping(ecephys_session=session)
    assert(rfm.name == 'Receptive Field Mapping')
    assert(set(rfm.unit_ids) == set(range(6)))
    assert(len(rfm.conditionwise_statistics) == 81*6)
    assert(rfm.conditionwise_psth.shape == (81, 249, 6))
    assert(not rfm.presentationwise_spike_times.empty)
    assert(len(rfm.presentationwise_statistics) == 81*6)
    assert(len(rfm.stimulus_conditions) == 81)


def test_stimulus(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    rfm = ReceptiveFieldMapping(ecephys_session=session)
    assert(isinstance(rfm.stim_table, pd.DataFrame))
    assert(len(rfm.stim_table) == 81)
    assert(set(rfm.stim_table.columns).issuperset({'x_position', 'y_position', 'start_time', 'stop_time'}))

    assert(set(rfm.azimuths) == {30.0, -20.0, 40.0, 20.0, 0.0, -30.0, -40.0, 10.0, -10.0})
    assert(rfm.number_azimuths == 9)

    assert(set(rfm.elevations) == {10.0, -10.0, 30.0, 40.0, -40.0, -30.0, -20.0, 20.0, 0.0})
    assert(rfm.number_elevations == 9)


def test_metrics(ecephys_api):
    session = EcephysSession(api=ecephys_api)
    rfm = ReceptiveFieldMapping(ecephys_session=session)
    assert(isinstance(rfm.metrics, pd.DataFrame))
    assert(len(rfm.metrics) == 6)
    assert(rfm.metrics.index.names == ['unit_id'])

    assert('azimuth_rf' in rfm.metrics.columns)
    assert('elevation_rf' in rfm.metrics.columns)
    assert('width_rf' in rfm.metrics.columns)
    assert('height_rf' in rfm.metrics.columns)
    assert('area_rf' in rfm.metrics.columns)
    assert('p_value_rf' in rfm.metrics.columns)
    assert('on_screen_rf' in rfm.metrics.columns)
    assert('firing_rate_rf' in rfm.metrics.columns)
    assert('fano_rf' in rfm.metrics.columns)
    assert('time_to_peak_rf' in rfm.metrics.columns)
    assert('reliability_rf' in rfm.metrics.columns)
    assert('lifetime_sparseness_rf' in rfm.metrics.columns)
    assert('run_pval_rf' in rfm.metrics.columns)
    assert('run_mod_rf' in rfm.metrics.columns)


@pytest.mark.skip(reason='Write a test for the receptive_fields()/_get_rf() methods')
def test_receptive_fields():
    # TODO: Implement
    pass


@pytest.mark.skip()
def test_response_by_stimulus_position():
    # TODO: Implement
    pass


@pytest.mark.skip()
def test_rf_stats():
    pass


@pytest.mark.skip()
def test_fit_2d_gaussian():
    pass

@pytest.mark.skip()
def test_invert_rf():
    pass

@pytest.mark.skip()
def test_threshold_rf():
    pass

if __name__ == '__main__':
    # test_load()
    # test_stimulus()
    test_metrics()
