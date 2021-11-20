"""
Created on Sunday July 15 2018

@author: marinag
"""
from itertools import chain
from typing import Dict, Any, Optional, List, Union
from allensdk.brain_observatory.behavior.sync.process_sync import (
    filter_digital, calculate_delay)  # NOQA: E402
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset  # NOQA: E402
import numpy as np
import scipy.stats as sps


def get_raw_stimulus_frames(
    dataset: SyncDataset, 
    permissive: bool = False
) -> np.ndarray:
    """ Report the raw timestamps of each stimulus frame. This corresponds to 
    the time at which the psychopy window's flip method returned, but not 
    necessarily to the time at which the stimulus frame was displayed.

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start).

    """
    try:
        return dataset.get_edges("falling",['stim_vsync', 'vsync_stim'], "seconds")
    except KeyError:
        if not permissive:
            raise
        return


def get_ophys_frames(
    dataset: SyncDataset, 
    permissive: bool = False
) -> np.ndarray:
    """ Report the timestamps of each optical physiology video frame

    Parameters
    ----------
    dataset : describes experiment timing

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start).
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Notes
    -----
    use rising edge for Scientifica, falling edge for Nikon 
    http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
    This function uses rising edges

    """
    try:
        print(dataset)
        return dataset.get_edges("rising", ['2p_vsync', 'vsync_2p'], "seconds")
    except KeyError:
        if not permissive:
            raise
        return


def get_lick_times(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Report the timestamps of each detected lick

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no lick timestamps were found in this sync 
        dataset.

    """
    return dataset.get_edges(
        "rising", ["lick_times", "lick_sensor"], "seconds", permissive)
    

def get_stim_photodiode(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[List[float]]:
    """ Report the timestamps of each detected sync square transition (both 
    black -> white and white -> black) in this experiment.

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no photodiode timestamps were found in this sync 
        dataset.

    """
    return dataset.get_edges(
        "all", ["stim_photodiode", "photodiode"], "seconds", permissive)


def get_trigger(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Returns (as a 1-element array) the time at which optical physiology 
    acquisition was started.

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

   Returns
    -------
    timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no timestamps were found in this sync dataset.

    Notes
    -----
    Ophys frame timestamps can be recorded before acquisition start when 
        experimenters are setting up the recording session. These do not 
        correspond to acquired ophys frames.

    """
    return dataset.get_edges(
        "rising", ["2p_trigger", "acq_trigger", "stim_running"], "seconds", permissive)


def get_eye_tracking(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Report the timestamps of each frame of the eye tracking video

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no eye tracking timestamps were found in this sync 
        dataset.

    """
    return dataset.get_edges(
        "rising", ["cam2_exposure", "eye_tracking", "eye_cam_exposing"], "seconds", permissive)


def get_behavior_monitoring(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Report the timestamps of each frame of the behavior 
    monitoring video

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no behavior monitoring timestamps were found in this 
        sync dataset.

    """
    return dataset.get_edges(
        "rising", ["cam1_exposure", "behavior_monitoring", "face_cam_exposing"], "seconds", 
        permissive)


def get_sync_data(
    sync_path: str,
    permissive: bool = False
) -> Dict[str, Union[List, np.ndarray, None]]:
    """ Convenience function for extracting several timestamp arrays from a 
    sync file.

    Parameters
    ----------
    sync_path : The hdf5 file here ought to be a Visual Behavior sync output 
        file. See allensdk.brain_observatory.sync_dataset for more details of 
        this format.
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised
    
    Returns
    -------
    A dictionary with the following keys. All timestamps in seconds:
        ophys_frames : timestamps of each optical physiology frame
        lick_times : timestamps of each detected lick
        ophys_trigger : The time at which ophys acquisition was started
        eye_tracking : timestamps of each eye tracking video frame
        behavior_monitoring : timestamps of behavior monitoring video frame
        stim_photodiode : timestamps of each photodiode transition
        stimulus_times_no_delay : raw stimulus frame timestamps
    Some values may be None. This indicates that the corresponding timestamps 
    were not located in this sync file.

    """

    sync_dataset = SyncDataset(sync_path)
    return {
        'ophys_frames': get_ophys_frames(sync_dataset, permissive),
        'lick_times': get_lick_times(sync_dataset, True),
        'ophys_trigger': get_trigger(sync_dataset, permissive),
        'eye_tracking': get_eye_tracking(sync_dataset, permissive),
        'behavior_monitoring': get_behavior_monitoring(sync_dataset, permissive),
        'stim_photodiode': get_stim_photodiode(sync_dataset, permissive),
        'stimulus_times_no_delay': get_raw_stimulus_frames(sync_dataset, permissive)
    }
{'ni_daq': {'device': 'Dev1', 'counter_output_freq': 100000.0,
 'sample_rate': 100000.0, 'counter_bits': 32, 'event_bits': 32}, 
 'start_time': '2021-09-29 10:52:09.689200', 'stop_time': '2021-09-29 12:07:58.586810',
  'line_labels': ['vsync_2p', '', 'vsync_stim', '', 'stim_photodiode', 'stim_running', '',
   '', 'beh_frame_received', 'eye_frame_received', 'face_frame_received', '', '', '', '', '',
    '', 'stim_running_opto', 'stim_trial_opto', '', '', 'beh_cam_frame_readout',
     'face_cam_frame_readout', '', '', 'eye_cam_frame_readout', '', 'beh_cam_exposing', 
     'face_cam_exposing', 'eye_cam_exposing', '', 'lick_sensor'], 'timeouts': [], 
     'version': '2.2.1+g1bc7438.b42257', 'sampling_type': 'frequency', 'file_version': '1.0.0', 
     'line_label_revision': 3, 'total_samples': 454880000}
{'total_samples': 483310000, 'sampling_type': 'frequency',
 'timeouts': [], 'start_time': '2018-11-08 11:21:58.256000', 
 'ni_daq': {'device': 'Dev1', 'event_bits': 32, 'counter_bits': 32,
  'sample_rate': 100000.0, 'counter_output_freq': 100000.0}, 
  'version': {'sync': 1.06, 'dataset': 1.04}, 'stop_time': '2018-11-08 12:42:31.568000',
   'line_labels': ['2p_vsync', '', 'stim_vsync', '', 'stim_photodiode', 'acq_trigger', '', '',
    'cam1_exposure', 'cam2_exposure', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
     '', '', '', '', '', '', '', 'lick_sensor']}