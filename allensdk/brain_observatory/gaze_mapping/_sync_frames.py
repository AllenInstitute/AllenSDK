import pandas as pd
from pathlib import Path

from allensdk.brain_observatory.sync_dataset import Dataset
from allensdk.brain_observatory import sync_utilities


def get_synchronized_camera_frame_times(session_sync_file: Path) -> pd.Series:
    """Get eye tracking camera frame times from an experiment session sync file.

    Args:
        session_sync_file (Path): Path to an ephys session sync file.
            The sync file contains rising/falling edges from a daq system which
            indicates when certain events occur (so they can be related to
            each other).

    Returns:
        pandas.Series: An array of times when frames for the eye tracking
            camera were acquired.
    """
    sync_dataset = Dataset(str(session_sync_file))

    frame_times = sync_dataset.get_edges(
        "rising", Dataset.EYE_TRACKING_KEYS, units="seconds"
    )

    # Occasionally an extra set of frame times are acquired after the rest of
    # the signals. We detect and remove these.
    frame_times = sync_utilities.trim_discontiguous_times(frame_times)

    return pd.Series(frame_times)
