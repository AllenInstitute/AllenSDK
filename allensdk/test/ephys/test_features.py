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
import pytest
import allensdk.ephys.ephys_features as ft
import numpy as np
import os
path = os.path.dirname(__file__)


def test_v_and_t_are_arrays():
    v = [0, 1, 2]
    t = [0, 1, 2]
    with pytest.raises(TypeError):
        ft.detect_putative_spikes(v, t)

    with pytest.raises(TypeError):
        ft.detect_putative_spikes(np.array(v), t)


def test_size_mismatch():
    v = np.array([0, 1, 2])
    t = np.array([0, 1])
    with pytest.raises(ft.FeatureError):
        ft.detect_putative_spikes(v, t)


def test_find_time_out_of_bounds():
    t = np.array([0, 1, 2])
    t_0 = 4

    with pytest.raises(ft.FeatureError):
        ft.find_time_index(t, t_0)


def test_dvdt_no_filter():
    t = np.array([0, 1, 2, 3])
    v = np.array([1, 1, 1, 1])

    assert np.allclose(ft.calculate_dvdt(v, t), np.diff(v) / np.diff(t))


def test_fixed_dt():
    t = [0, 1, 2, 3]
    assert ft.has_fixed_dt(t) == True

    # Change the first time point to make time steps inconsistent
    t[0] -= 3.
    assert ft.has_fixed_dt(t) == False


def test_detect_one_spike():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    expected_spikes = np.array([728])

    assert np.allclose(ft.detect_putative_spikes(v[:3000], t[:3000]), expected_spikes)


def test_detect_two_spikes():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    expected_spikes = np.array([728, 3386])

    assert np.allclose(ft.detect_putative_spikes(v, t), expected_spikes)


def test_detect_no_spikes():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = np.zeros_like(t)

    assert len(ft.detect_putative_spikes(v, t)) == 0


def test_detect_no_spike_peaks():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = np.zeros_like(t)
    spikes = np.array([])

    assert len(ft.find_peak_indexes(v, t, spikes)) == 0


def test_detect_two_spike_peaks():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([728, 3386])
    expected_peaks = np.array([812, 3478])

    assert np.allclose(ft.find_peak_indexes(v, t, spikes), expected_peaks)


def test_filter_problem_spikes():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([728, 3386])
    peaks = np.array([812, 3478])

    new_spikes, new_peaks = ft.filter_putative_spikes(v, t, spikes, peaks)
    assert np.allclose(spikes, new_spikes)
    assert np.allclose(peaks, new_peaks)


def test_filter_no_spikes():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = np.zeros_like(t)
    spikes = np.array([])
    peaks = np.array([])

    new_spikes, new_peaks = ft.filter_putative_spikes(v, t, spikes, peaks)
    assert len(new_spikes) == len(new_peaks) == 0


def test_upstrokes():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([728, 3386])
    peaks = np.array([812, 3478])

    expected_upstrokes = np.array([778, 3440])
    assert np.allclose(ft.find_upstroke_indexes(v, t, spikes, peaks), expected_upstrokes)


def test_thresholds():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    upstrokes = np.array([778, 3440])

    expected_thresholds = np.array([725, 3382])
    assert np.allclose(ft.refine_threshold_indexes(v, t, upstrokes), expected_thresholds)


def test_thresholds_cannot_find_target():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    upstrokes = np.array([778, 3440])

    expected = np.array([0, 778])
    assert np.allclose(ft.refine_threshold_indexes(v, t, upstrokes, thresh_frac=-5.0), expected)


def test_check_spikes_and_peaks():
    t = np.arange(0, 30) * 5e-6
    v = np.zeros_like(t)
    spikes = np.array([0, 5])
    peaks = np.array([10, 15])
    upstrokes = np.array([3, 13])

    new_spikes, new_peaks, new_upstrokes, clipped = ft.check_thresholds_and_peaks(v, t, spikes, peaks, upstrokes)
    assert np.allclose(new_spikes, spikes[:-1])
    assert np.allclose(new_peaks, peaks[1:])


def test_troughs():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([725, 3382])
    peaks = np.array([812, 3478])

    expected_troughs = np.array([1089, 3741])
    assert np.allclose(ft.find_trough_indexes(v, t, spikes, peaks), expected_troughs)


def test_troughs_with_peak_at_end():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([725, 3382])
    peaks = np.array([812, 3478])
    clipped = np.array([False, True])

    troughs = ft.find_trough_indexes(v[:peaks[-1]], t[:peaks[-1]],
                                     spikes, peaks, clipped=clipped)
    assert np.isnan(troughs[-1])


def test_downstrokes():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    peaks = np.array([812, 3478])
    troughs = np.array([1089, 3741])

    expected_downstrokes = np.array([862, 3532])
    assert np.allclose(ft.find_downstroke_indexes(v, t, peaks, troughs), expected_downstrokes)


def test_downstrokes_too_many_troughs():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    peaks = np.array([812, 3478])
    troughs = np.array([1089, 3741, 3999])

    with pytest.raises(ft.FeatureError):
        ft.find_downstroke_indexes(v, t, peaks, troughs)


def test_width_calculation():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([725, 3382])
    peaks = np.array([812, 3478])
    troughs = np.array([1089, 3741])

    expected_widths = np.array([0.000545, 0.000585])
    assert np.allclose(ft.find_widths(v, t, spikes, peaks, troughs), expected_widths)


def test_width_calculation_too_many_troughs():
    data = np.loadtxt(os.path.join(path, "data/spike_test_pair.txt"))
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([725, 3382])
    peaks = np.array([812, 3478])
    troughs = np.array([1089, 3741, 3999])

    with pytest.raises(ft.FeatureError):
        ft.find_widths(v, t, spikes, peaks, troughs)


@pytest.mark.skipif(True, reason="not implemented")
def test_width_calculation_with_burst():
    # example sp 487663469, sweep 43
    pass
