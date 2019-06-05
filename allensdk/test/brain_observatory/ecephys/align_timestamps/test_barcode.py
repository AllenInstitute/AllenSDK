import pytest
import numpy as np
import pandas as pd

import allensdk.brain_observatory.ecephys.align_timestamps.barcode as barcode


@pytest.fixture
def two_barcodes():

    on_times = np.array([11, 14, 30, 32, 34])
    off_times = np.array([13, 15, 31, 33, 35])

    ibi = 10
    bar_duration = 1.0
    bar_duration_ceiling = 7
    nbits = 4

    return on_times, off_times, ibi, bar_duration, bar_duration_ceiling, nbits


@pytest.fixture
def master_barcodes_sequence():

    master_times = np.array([10, 25, 30, 37, 44, 45])
    master_barcodes = np.array([1, 2, 3, 4, 5, 6])

    return master_times, master_barcodes


def test_extract_barcodes_from_times(two_barcodes):

    starts_obt, codes_obt = barcode.extract_barcodes_from_times(*two_barcodes)

    starts_exp = [30]
    codes_exp = [5]

    assert np.allclose(starts_obt, starts_exp)
    assert np.allclose(codes_obt, codes_exp)


@pytest.mark.parametrize("sc", [1.0])  # 0.5, 10, .3, -14])
@pytest.mark.parametrize("tr", [-3])  # 22, -11])
@pytest.mark.parametrize("sind", [0])  # , 0, -5, 4.3])
@pytest.mark.parametrize("prate", [10])  # , 1, -7, 0.1])
@pytest.mark.parametrize("npcodes", [-1])  # , 3])
def test_get_time_offset(sc, tr, sind, prate, npcodes, master_barcodes_sequence):

    master_times, master_barcodes = master_barcodes_sequence
    probe_times = (master_times[:npcodes] + tr) * sc
    probe_barcodes = master_barcodes[:npcodes]

    obt = barcode.get_probe_time_offset(
        master_times, master_barcodes, probe_times, probe_barcodes, sind, prate
    )
    obt = [obt[0][0], obt[1][0], (obt[2][0][0], obt[2][1][0])]

    # total_time_shift, probe_rate, master_endpoints
    exp = [
        tr - sind / (sc * prate),
        sc * prate,
        (master_times[0], master_times[npcodes - 1]),
    ]

    for exp_el, obt_el in zip(exp, obt):
        assert np.allclose(exp_el, obt_el)


@pytest.mark.parametrize("sc", [-10, -2, -1, -0.5, 0.5, 1, 2, 10])
@pytest.mark.parametrize("tr", [-10, -2, -1, -0.5, 0, 0.5, 1, 2, 10])
def test_linear_transform_from_intervals(sc, tr, master_barcodes_sequence):

    master = np.array([1, 2])
    probe = (master + tr) * sc

    sc_obt, tr_obt = barcode.linear_transform_from_intervals(master, probe)

    assert sc == sc_obt
    assert tr == tr_obt


@pytest.mark.parametrize("sc", [-10, -2, -1, -0.5, 0.5, 1, 2, 10])
@pytest.mark.parametrize("tr", [-10, -2, -1, -0.5, 0, 0.5, 1, 2, 10])
@pytest.mark.parametrize(
    "npcodes", [-1, 3]
)  # fails in region [0, 2] due to insufficient samples
def test_match_barcodes(sc, tr, npcodes, master_barcodes_sequence):

    master_times, master_barcodes = master_barcodes_sequence
    probe_times = (master_times + tr) * sc
    probe_times_cut = probe_times[:npcodes]
    probe_barcodes = master_barcodes[:npcodes]

    pint, mint = barcode.match_barcodes(
        master_times, master_barcodes, probe_times_cut, probe_barcodes
    )
    assert pint[1] - pint[0] == sc * (mint[1] - mint[0])
