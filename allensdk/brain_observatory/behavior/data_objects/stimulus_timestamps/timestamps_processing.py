from typing import Union
from pathlib import Path

import numpy as np

from allensdk.internal.brain_observatory.time_sync import OphysTimeAligner


def get_behavior_stimulus_timestamps(stimulus_pkl: dict) -> np.ndarray:
    """Obtain visual behavior stimuli timing information from a behavior
    stimulus *.pkl file.

    Parameters
    ----------
    stimulus_pkl : dict
        A dictionary containing stimulus presentation timing information
        during a behavior session. Presentation timing info is stored as
        an array of times between frames (frame time intervals) in
        milliseconds.

    Returns
    -------
    np.ndarray
        Timestamps (in seconds) for presented stimulus frames during a session.
    """
    vsyncs = stimulus_pkl["items"]["behavior"]["intervalsms"]
    stimulus_timestamps = np.hstack((0, vsyncs)).cumsum() / 1000.0
    return stimulus_timestamps


def get_ophys_stimulus_timestamps(sync_path: Union[str, Path]) -> np.ndarray:
    """Obtain visual behavior stimuli timing information from a sync *.h5 file.

    Parameters
    ----------
    sync_path : Union[str, Path]
        The path to a sync *.h5 file that contains global timing information
        about multiple data streams (e.g. behavior, ophys, eye_tracking)
        during a session.

    Returns
    -------
    np.ndarray
        Timestamps (in seconds) for presented stimulus frames during a
        behavior + ophys session.
    """
    aligner = OphysTimeAligner(sync_file=sync_path)
    stimulus_timestamps, _ = aligner.clipped_stim_timestamps
    return stimulus_timestamps
