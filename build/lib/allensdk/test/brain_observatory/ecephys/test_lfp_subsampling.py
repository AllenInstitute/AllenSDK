import pytest
import itertools
import numpy as np
import logging

import allensdk.brain_observatory.ecephys.lfp_subsampling.subsampling as subsampling


@pytest.mark.parametrize('total_channels', [100, 384])
@pytest.mark.parametrize('surface_offset', [-20, -50])
@pytest.mark.parametrize('surface_padding', [10, 20])
@pytest.mark.parametrize('start_channel_offset', [0, 1, 2])
@pytest.mark.parametrize('channel_stride', [1, 2, 4, 10])
def test_select_channels(total_channels, surface_offset, surface_padding, start_channel_offset, channel_stride):
    input_channels = np.arange(start_channel_offset, total_channels + surface_offset + surface_padding)
    selected, actual = subsampling.select_channels(total_channels=total_channels,
                                                   surface_channel=total_channels + surface_offset,
                                                   surface_padding=surface_padding,
                                                   start_channel_offset=start_channel_offset,
                                                   channel_stride=channel_stride,
                                                   channel_order=np.arange(total_channels))

    assert np.allclose(selected, actual)
    assert len(selected) == len(input_channels[::channel_stride])


@pytest.mark.parametrize('remove_references', [True, False])
@pytest.mark.parametrize('reference_channels', [np.array([0, 1, 2]), np.array([9, 10, 11]), np.array([10, 11, 12])])
@pytest.mark.parametrize('remove_noisy_channels', [True, False])
@pytest.mark.parametrize('noisy_channels', [np.array([10, 11, 12])])
def test_select_channels_filtered(remove_references, reference_channels, remove_noisy_channels, noisy_channels):
    """Similar to test above but focused on ability to remove reference """
    total_channels = 100
    surface_offset = -20
    start_channel_offset = 0
    channel_stride = 1
    surface_padding = 10

    selected, actual = subsampling.select_channels(total_channels=total_channels,
                                                   surface_channel=total_channels + surface_offset,
                                                   surface_padding=surface_padding,
                                                   start_channel_offset=start_channel_offset,
                                                   channel_stride=channel_stride,
                                                   channel_order=np.arange(total_channels),
                                                   noisy_channels=noisy_channels,
                                                   remove_noisy_channels=remove_noisy_channels,
                                                   reference_channels=reference_channels,
                                                   remove_references=remove_references)

    assert np.allclose(selected, actual)
    removed_channels = set()
    if remove_noisy_channels:
        assert(not np.any(np.isin(noisy_channels, selected)))
        removed_channels |= set(noisy_channels)

    if remove_references:
        assert(not np.any(np.isin(reference_channels, selected)))
        removed_channels |= set(reference_channels)

    input_channels = np.arange(start_channel_offset, total_channels + surface_offset + surface_padding)
    assert(len(selected) == len(input_channels) - len(removed_channels))


@pytest.mark.parametrize('array_length', [50])  # , 150, 2001])
@pytest.mark.parametrize('subsampling_factor', [1])  # , 2, 4, 10])
def test_subsample_timestamps(subsampling_factor, array_length):
    timestamps = np.linspace(0, 50, array_length)
    ts_subsampled = subsampling.subsample_timestamps(timestamps, subsampling_factor)

    assert len(ts_subsampled) == np.ceil(len(timestamps) / subsampling_factor)


def test_subsample_lfp():
    lfp_raw = np.zeros((100, 100))
    selected_channels = np.arange(0, 50, 5)
    subsampling_factor = 2
    lfp_subsampled = subsampling.subsample_lfp(lfp_raw, selected_channels, subsampling_factor)

    assert lfp_subsampled.shape == (50, 10)


def test_remove_lfp_offset():
    lfp_raw = np.zeros((2500, 100)) + 10
    lfp_filtered = subsampling.remove_lfp_offset(lfp_raw, 2500.0, 0.1, 1)

    assert np.max(lfp_filtered) < 1e-10


def test_remove_lfp_noise():
    lfp_raw = np.zeros((2500, 100))
    lfp_raw[:, -10:] = 1
    channel_numbers = np.arange(100)
    lfp_noise_removed = subsampling.remove_lfp_noise(lfp_raw, 90, channel_numbers)

    # TODO: This is not safe, try using set to assure that the removed noise contains only -1 and 0's
    assert np.array_equal(np.unique(lfp_noise_removed), np.array([-1, 0]))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger('ecephys_pipeline.modules.lfp_subsampling').setLevel(logging.INFO)
    for tc, so, sp, sco, cs in itertools.product([100, 384], [-20, -50], [10, 20], [0, 1, 2], [1, 2, 4, 10]):
        test_select_channels(total_channels=tc, surface_offset=so, surface_padding=sp, start_channel_offset=sco,
                             channel_stride=cs)
    test_subsample_timestamps(subsampling_factor=1, array_length=50)
    test_subsample_lfp()
    test_remove_lfp_offset()
    test_remove_lfp_noise()
