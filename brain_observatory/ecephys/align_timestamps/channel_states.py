import numpy as np

from . import barcode


def extract_barcodes_from_states(
    channel_states, timestamps, sampling_rate, **barcode_kwargs
):
    """Obtain barcodes from timestamped rising/falling edges.

    Parameters
    ----------
    channel_states : numpy.ndarray
        Rising and falling edges, denoted 1 and -1
    timestamps : numpy.ndarray
        Sample index of each event.
    sampling_rate : numeric
        Samples / second
    **barcode_kwargs : 
        Additional parameters describing the barcodes.


    """

    on_events = np.where(channel_states == 1)
    off_events = np.where(channel_states == -1)

    T_on = timestamps[on_events] / float(sampling_rate)
    T_off = timestamps[off_events] / float(sampling_rate)

    return barcode.extract_barcodes_from_times(T_on, T_off, **barcode_kwargs)


def extract_splits_from_states(
    channel_states, timestamps, sampling_rate, **barcode_kwargs
):
    """Obtain barcodes from timestamped rising/falling edges.

    Parameters
    ----------
    channel_states : numpy.ndarray
        Rising and falling edges, denoted 1 and -1
    timestamps : numpy.ndarray
        Sample index of each event.
    sampling_rate : numeric
        Samples / second
    **barcode_kwargs : 
        Additional parameters describing the barcodes.


    """

    split_events = np.where(channel_states == 0)

    T_split = timestamps[split_events] / float(sampling_rate)

    if len(T_split) == 0:
        T_split = np.array([0])

    return T_split
