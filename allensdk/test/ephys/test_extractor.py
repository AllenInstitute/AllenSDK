import pytest
import numpy as np
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor, input_resistance
import os
path = os.path.dirname(__file__)


def test_extractor_no_values():
    ext = EphysSweepSetFeatureExtractor()


def test_extractor_wrong_inputs():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    i = np.zeros_like(v)

    with pytest.raises(ValueError):
        ext = EphysSweepSetFeatureExtractor(t, v, i)

    with pytest.raises(ValueError):
        ext = EphysSweepSetFeatureExtractor([t], v, i)

    with pytest.raises(ValueError):
        ext = EphysSweepSetFeatureExtractor([t], [v], i)

    with pytest.raises(ValueError):
        ext = EphysSweepSetFeatureExtractor([t, t], [v], [i])

    with pytest.raises(ValueError):
        ext = EphysSweepSetFeatureExtractor([t, t], [v, v], [i])


def test_extractor_on_sample_data():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]

    ext = EphysSweepSetFeatureExtractor([t], [v])
    ext.process_spikes()
    swp = ext.sweeps()[0]
    spikes = swp.spikes()

    keys = swp.spike_feature_keys()
    swp_keys = swp.sweep_feature_keys()
    result = swp.spike_feature(keys[0])
    result = swp.sweep_feature("first_isi")
    result = ext.sweep_features("first_isi")
    result = ext.spike_feature_averages(keys[0])

    with pytest.raises(KeyError):
        result = swp.spike_feature("nonexistent_key")

    with pytest.raises(KeyError):
        result = swp.sweep_feature("nonexistent_key")


def test_extractor_on_sample_data_with_i():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    i = np.zeros_like(v)

    ext = EphysSweepSetFeatureExtractor([t], [v], [i])
    ext.process_spikes()


def test_extractor_on_zero_voltage():
    t = np.arange(0, 4000) * 5e-5
    v = np.zeros_like(t)
    i = np.zeros_like(t)

    ext = EphysSweepSetFeatureExtractor([t], [v], [i])
    ext.process_spikes()


def test_extractor_on_variable_time_step():
    data = np.loadtxt(os.path.join(path, "data/spike_test_var_dt.txt"))
    t = data[:, 0]
    v = data[:, 1]

    ext = EphysSweepSetFeatureExtractor([t], [v])
    ext.process_spikes()
    expected_thresh_ind = np.array([73, 183, 314, 463, 616, 770])
    sweep = ext.sweeps()[0]
    assert np.allclose(sweep.spike_feature("threshold_index"), expected_thresh_ind)


def test_extractor_with_high_init_dvdt():
    data = np.loadtxt(os.path.join(path, "data/spike_test_high_init_dvdt.txt"))
    t = data[:, 0]
    v = data[:, 1]

    ext = EphysSweepSetFeatureExtractor([t], [v])
    ext.process_spikes()
    expected_thresh_ind = np.array([11222, 16256, 24058])
    sweep = ext.sweeps()[0]
    assert np.allclose(sweep.spike_feature("threshold_index"), expected_thresh_ind)


def test_extractor_input_resistance():
    t = np.arange(0, 1.0, 5e-6)
    v1 = np.ones_like(t) * -5.
    v2 = np.ones_like(t) * -10.
    i1 = np.ones_like(t) * -50.
    i2 = np.ones_like(t) * -100.

    ext = EphysSweepSetFeatureExtractor([t, t], [v1, v2], [i1, i2])
    ri = input_resistance(ext)
    assert np.allclose(ri, 100.)
