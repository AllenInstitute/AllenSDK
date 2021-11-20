# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import mock

import pytest
import numpy as np
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor, input_resistance
import allensdk.ephys.ephys_extractor as ephys_extractor
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
    t = np.arange(0, 4000) * 5e-6
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
    expected_thresh_ind = np.array([11222, 16258, 24060])
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


def test_fit_fi_slope():

    nsweeps = 5
    weights = np.array([ 2, 1 ])

    amps = np.random.rand(nsweeps)
    iteramps = iter(amps)

    design = np.array([amps, np.ones_like(amps)]).T
    rates = np.dot(design, weights)
    build_stim_amps = lambda: lambda sweep: next(iteramps)

    class Ext(object):
        def sweeps(self):
            return np.zeros([nsweeps])
        def sweep_features(self, key):
            return rates

    with mock.patch(
        'allensdk.ephys.ephys_extractor._step_stim_amp', 
        new_callable=build_stim_amps) as p:

        slope_obt = ephys_extractor.fit_fi_slope(Ext())
        assert(np.allclose(weights[0], slope_obt))