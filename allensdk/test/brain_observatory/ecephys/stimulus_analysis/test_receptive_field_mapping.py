import pytest
import pandas as pd
import numpy as np

from .conftest import MockSessionApi
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import \
    ReceptiveFieldMapping, \
    fit_2d_gaussian, \
    threshold_rf

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
    rfm = ReceptiveFieldMapping(ecephys_session=session, minimum_spike_count=1.0, trial_duration=0.25,
                                mask_threshold=0.5)
    assert(isinstance(rfm.metrics, pd.DataFrame))
    assert(len(rfm.metrics) == 6)
    assert(rfm.metrics.index.names == ['unit_id'])

    # TODO: Methods are too sensitive and will have different values depending on the version of scipy
    assert('azimuth_rf' in rfm.metrics.columns)
    assert('elevation_rf' in rfm.metrics.columns)
    assert('width_rf' in rfm.metrics.columns)
    assert('height_rf' in rfm.metrics.columns)
    # Different versions of scipy will return unit 1 as either a 0.0 or a nan
    #assert(np.allclose(rfm.metrics['height_rf'].loc[[0, 1, 2, 3, 4, 5]],
    #                   [np.nan, 0.0, 129.522395, np.nan, np.nan, np.nan], equal_nan=True))

    assert('area_rf' in rfm.metrics.columns)
    assert(np.allclose(rfm.metrics['area_rf'].loc[[0, 1, 2, 3, 4, 5]],
                       [0.0, 0.0, 0.0, np.nan, 0.0, 0.0], equal_nan=True))

    assert('p_value_rf' in rfm.metrics.columns)
    assert('on_screen_rf' in rfm.metrics.columns)
    assert('firing_rate_rf' in rfm.metrics.columns)
    assert('fano_rf' in rfm.metrics.columns)
    assert('time_to_peak_rf' in rfm.metrics.columns)
    assert('lifetime_sparseness_rf' in rfm.metrics.columns)
    assert('run_pval_rf' in rfm.metrics.columns)
    assert('run_mod_rf' in rfm.metrics.columns)


def test_receptive_fields(ecephys_api):
    # Also test_response_by_stimulus_position()
    session = EcephysSession(api=ecephys_api)
    rfm = ReceptiveFieldMapping(ecephys_session=session)
    assert(rfm.receptive_fields)
    assert(type(rfm.receptive_fields))
    assert('spike_counts' in rfm.receptive_fields)
    assert(rfm.receptive_fields['spike_counts'].shape == (9, 9, 6))  # x, y, units
    assert(set(rfm.receptive_fields['spike_counts'].coords) == {'y_position', 'x_position', 'unit_id'})
    assert(np.all(rfm.receptive_fields['spike_counts'].coords['x_position']
                  == [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]))
    assert(np.all(rfm.receptive_fields['spike_counts'].coords['y_position']
                  == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]))

    # Some randomly sampled testing to make sure everything works like it should
    assert(rfm.receptive_fields['spike_counts'][{'unit_id': 0}].values.sum() == 4)
    assert(rfm.receptive_fields['spike_counts'][{'unit_id': 3}].values.sum() == 0)
    assert(rfm.receptive_fields['spike_counts'][{'unit_id': 2, 'x_position': 8, 'y_position': 3}] == 3)
    assert(np.all(rfm.receptive_fields['spike_counts'][{'x_position': 2, 'y_position': 5}] == [1, 0, 0, 0, 1, 1]))



##  Some special receptive fields for testing

# Data taken from real example
rf_field_real = np.array([[7440, 5704,  11408, 8184, 9920, 5952, 11904, 11904, 9672],
                          [8184, 12152, 10912, 12648, 15128, 19096, 17112, 14384, 11656],
                          [12152, 17856, 25048, 36208, 47368, 30256, 20336, 10912, 10168],
                          [15624, 31000, 53568, 92752, 119288, 69440, 31496, 16120, 10416],
                          [12152, 23560, 32984, 74896, 93496, 52328, 28024, 19592, 11656],
                          [9672, 7192, 10912, 16120, 16368, 18600, 14880, 6696, 11408],
                          [11656, 7688, 6696, 5456, 11408, 9672, 11160, 12152, 7936],
                          [6696, 6696, 9424, 8928, 6200, 11160, 7688, 6200, 9672],
                          [8928, 10912, 9176, 8432, 7688, 9424, 5704, 8184, 14384]], dtype=np.float64)

# RF as a typical gaussian
x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
rf_field_gaussian = np.exp(-((np.sqrt(x*x + y*y) - 0.0)**2 /(2.0*1.0**2)))

# Only activity at one of the corners of the field
rf_field_edge = np.zeros((9, 9))
rf_field_edge[8, 8] = 5.0


@pytest.mark.parametrize('rf,threshold,expected_mask,expected_x,expected_y,expected_area',
                         [
                             (np.zeros((9, 9)), 0.5, np.zeros((9, 9)), np.nan, np.nan, 0.0), # No firing
                             (np.full((9, 9), 100.0), 0.5, np.zeros((9, 9)), np.nan, np.nan, 0.0),  # completely consistant firing, no center
                             (rf_field_real, 0.5, None, 3.5, 3.0, 2.0),  # example from real data
                             (rf_field_gaussian, 0.5, None, 4.0, 4.0, 9.0),
                             (rf_field_edge, 0.05, None, 8.0, 8.0, 1.0)
                         ])
def test_threshold_rf(rf, threshold, expected_mask, expected_x, expected_y, expected_area):
    mask_rf, x, y, area = threshold_rf(rf, threshold)
    assert(np.isclose(x, expected_x, equal_nan=True))
    assert(np.isclose(y, expected_y, equal_nan=True))
    assert(np.isclose(area, expected_area, equal_nan=True))
    if expected_mask is not None:
        # TODO: Find a better way to check the resulting mask, it should match up with the center/area
        assert(np.allclose(mask_rf, expected_mask, equal_nan=True))


@pytest.mark.parametrize('matrix,expected',
                         [
                             (rf_field_real, (np.array([1.04991433e+05, 3.74217858e+00, 3.24465965e+00, 1.66477569e+00, 1.04485211e+00]), True)),
                             (rf_field_gaussian, (np.array([1.0, 4.0, 4.0, 4.0, 4.0]), True)),
                             (np.zeros((9, 9)), ((np.nan, np.nan, np.nan, np.nan, np.nan), False)),
                             ## These edge cases are too sensitive and will produce different values depending on the
                             ## version of scipy is compiled against.
                             # (np.full((9, 9), 20.5), (np.array([20.5000000, 3.62601891, 3.55521927, 1.20266006e+05, 1.08161135e+05]), True)),
                             # (rf_field_edge, (np.array([5.0, 8.0, 8.0, 0.0, 0.0]), True))
                         ])
def test_fit_2d_gaussian(matrix, expected):
    fit_params, success = fit_2d_gaussian(matrix)
    assert(np.allclose(fit_params, expected[0], equal_nan=True))
    assert(success == expected[1])


if __name__ == '__main__':
    # test_load()
    # test_stimulus()
    test_metrics()
    # test_receptive_fields()

    # test_threshold_rf(np.zeros((9, 9)), 0.5, np.zeros((9, 9)), np.nan, np.nan, 0.0)  # No firing
    # test_threshold_rf(np.full((9, 9), 100.0), 0.5, np.zeros((9, 9)), np.nan, np.nan, 0.0)  # completely consistant firing, no center
    # test_threshold_rf(rf_field_real, 0.5, None, 3.5, 3.0, 2.0)  # example from real data
    # test_threshold_rf(rf_field_gaussian, 0.5, None, 4.0, 4.0, 9.0)
    # test_threshold_rf(rf_field_edge, 0.05, None, 8.0, 8.0, 1.0)

    # test_fit_2d_gaussian(rf_field_real, (np.array([1.04991433e+05, 3.74217858e+00, 3.24465965e+00, 1.66477569e+00, 1.04485211e+00]), True))
    # test_fit_2d_gaussian(rf_field_gaussian, (np.array([1.0, 4.0, 4.0, 4.0, 4.0]), True))
    # test_fit_2d_gaussian(np.zeros((9, 9)), ((np.nan, np.nan, np.nan, np.nan, np.nan), False))
    # test_fit_2d_gaussian(np.full((9, 9), 20.5), (np.array([20.5000000, 3.62601891, 3.55521927, 1.20266006e+05, 1.08161135e+05]), True))
    test_fit_2d_gaussian(rf_field_edge, (np.array([5.0, 8.0, 8.0, 0.0, 0.0]), True))
    pass
