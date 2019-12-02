from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from allensdk.brain_observatory.sync_dataset import Dataset


def trim_discontiguous_times(times, threshold=100):
    times = np.array(times)
    intervals = np.diff(times)

    med_interval = np.median(intervals)
    interval_threshold = med_interval * threshold

    gap_indices = np.where(intervals > interval_threshold)[0]

    if len(gap_indices) == 0:
        return times

    return times[:gap_indices[0] + 1]


def get_synchronized_frame_times(session_sync_file: Path,
                                 sync_line_label_keys: Tuple[str, ...]) -> pd.Series:
    """Get experimental frame times from an experiment session sync file.

    Parameters
    ----------
    session_sync_file : Path
        Path to an ephys session sync file.
        The sync file contains rising/falling edges from a daq system which
        indicates when certain events occur (so they can be related to
        each other).
    sync_line_label_keys : Tuple[str, ...]
        Line label keys to get times for. See class attributes of
        allensdk.brain_observatory.sync_dataset.Dataset for a listing of
        possible keys.

    Returns
    -------
    pd.Series
        An array of times when frames for the eye tracking camera were acquired.
    """
    sync_dataset = Dataset(str(session_sync_file))

    frame_times = sync_dataset.get_edges(
        "rising", sync_line_label_keys, units="seconds"
    )

    # Occasionally an extra set of frame times are acquired after the rest of
    # the signals. We detect and remove these.
    frame_times = trim_discontiguous_times(frame_times)

    return pd.Series(frame_times)
