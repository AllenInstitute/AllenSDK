import numpy as np
import h5py
from allensdk.brain_observatory.sync_dataset import Dataset
import pandas as pd
import logging
try:
    import cv2
except ImportError:
    cv2 = None

TRANSITION_FRAME_INTERVAL = 60
SHORT_PHOTODIODE_MIN = 0.1 # seconds
SHORT_PHOTODIODE_MAX = 0.5 # seconds
REG_PHOTODIODE_MIN = 1.9 # seconds
REG_PHOTODIODE_MAX = 2.1 # seconds
PHOTODIODE_ANOMALY_THRESHOLD = 0.5 # seconds
LONG_STIM_THRESHOLD = 0.2 # seconds
ASSUMED_DELAY = 0.0351 # seconds
MAX_MONITOR_DELAY = 0.07 # seconds

VERSION_1_KEYS = {
    "photodiode": "stim_photodiode",
    "2p": "2p_vsync",
    "stimulus": "stim_vsync",
    "eye_camera": "cam2_exposure",
    "behavior_camera": "cam1_exposure",
    "acquiring": "2p_acquiring",
    "lick_sensor": "lick_1"
    }

# MPE is changing keys. This isn't versioned in the file.
VERSION_2_KEYS = {
    "photodiode": "photodiode",
    "2p": "2p_vsync",
    "stimulus": "stim_vsync",
    "eye_camera": "eye_tracking",
    "behavior_camera": "behavior_monitoring",
    "acquiring": "2p_acquiring",
    "lick_sensor": "lick_sensor"
    }


def get_keys(sync_dset):
    """Get the correct lookup for line labels.

    This method is fragile, but not all old data contains the full list
    of keys.
    """
    if "cam2_exposure" in sync_dset.line_labels:
        return VERSION_1_KEYS
    return VERSION_2_KEYS


def monitor_delay(sync_dset, stim_times, photodiode_key,
                  transition_frame_interval=TRANSITION_FRAME_INTERVAL,
                  max_monitor_delay=MAX_MONITOR_DELAY,
                  assumed_delay=ASSUMED_DELAY):
    """Calculate monitor delay."""
    try:
        transitions = stim_times[::transition_frame_interval]
        photodiode_events = get_real_photodiode_events(sync_dset, photodiode_key)
        transition_events = photodiode_events[0:len(transitions)]

        delay = np.mean(transition_events-transitions)
        logging.info("Calculated monitor delay: %s", delay)

        if delay < 0 or delay > max_monitor_delay:
            delay = assumed_delay
            logging.warning("Setting delay to assumed value: %s", delay)
    except (IndexError, ValueError) as e:
        logging.error(e)
        delay = assumed_delay
        logging.warning("Bad photodiode signal, setting delay to assumed "
                        "value: %s", delay)

    return delay


def get_photodiode_events(sync_dset, photodiode_key):
    """Returns the photodiode events with the start/stop indicators and
    the window init flash stripped off.
    """
    all_events = sync_dset.get_events_by_line(photodiode_key)
    pdr = sync_dset.get_rising_edges(photodiode_key)
    pdf = sync_dset.get_falling_edges(photodiode_key)

    all_events_sec = all_events/sync_dset.sample_freq
    pdr_sec = pdr/sync_dset.sample_freq
    pdf_sec = pdf/sync_dset.sample_freq

    pdf_diff = np.ediff1d(pdf_sec, to_end=0)
    pdr_diff = np.ediff1d(pdr_sec, to_end=0)

    reg_pd_falling = pdf_sec[(pdf_diff >= REG_PHOTODIODE_MIN) &
                             (pdf_diff <= REG_PHOTODIODE_MAX)]

    short_pd_rising = pdr_sec[(pdr_diff >= SHORT_PHOTODIODE_MIN) &
                              (pdr_diff <= SHORT_PHOTODIODE_MAX)]

    first_falling = reg_pd_falling[0]
    last_falling = reg_pd_falling[-1]

    end_indicators = short_pd_rising[short_pd_rising > last_falling]
    first_end_indicator = end_indicators[0]

    pd_events =  all_events_sec[(all_events_sec >= first_falling) &
                                (all_events_sec < first_end_indicator)]
    return pd_events


def get_real_photodiode_events(sync_dset, photodiode_key,
                               anomaly_threshold=PHOTODIODE_ANOMALY_THRESHOLD):
    """Gets the photodiode events with the anomalies removed."""
    events = get_photodiode_events(sync_dset, photodiode_key)
    anomalies = np.where(np.diff(events) < anomaly_threshold)
    return np.delete(events, anomalies)


def get_alignment_array(ref, other, int_method=np.floor):
    """Generate an alignment array """
    return int_method(np.interp(other, ref, np.arange(len(ref)), left=np.nan,
                      right=np.nan))


def get_video_length(filename):
    if cv2 is not None:
        try:
            capture = cv2.VideoCapture(filename)
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except AttributeError:
            logging.warning("Could not get length for %s, opencv out of date",
                            filename)
    else:
        logging.warning("Could not get length for %s", filename)


def get_ophys_data_length(filename):
    with h5py.File(filename) as f:
        return f["data"].shape[1]


def get_stim_data_length(filename: str) -> int:
    """Get stimulus data length from .pkl file.

    Parameters
    ----------
    filename : str
        Path of stimulus data .pkl file.

    Returns
    -------
    int
        Stimulus data length.
    """
    stim_data = pd.read_pickle(filename)

    # A subset of stimulus .pkl files do not have the "vsynccount" field.
    # MPE *won't* be backfilling the "vsynccount" field for these .pkl files.
    # So the least worst option is to recalculate the vsync_count.
    try:
        vsync_count = stim_data["vsynccount"]
    except KeyError:
        vsync_count = len(stim_data["items"]["behavior"]["intervalsms"]) + 1

    return vsync_count


def corrected_video_timestamps(video_name, timestamps, data_length):
    delta = 0
    if data_length is not None:
        delta = len(timestamps) - data_length
        if delta != 0:
            logging.info("%s data of length %s has timestamps of length "
                         "%s", video_name, data_length, len(timestamps))
    else:
        logging.info("No data length provided for %s", video_name)
    
    return timestamps, delta


class OphysTimeAligner(object):
    def __init__(self, sync_file, scanner=None, dff_file=None,
                 stimulus_pkl=None, eye_video=None, behavior_video=None,
                 long_stim_threshold=LONG_STIM_THRESHOLD):
        self.scanner = scanner if scanner is not None else "SCIVIVO"
        self._dataset = Dataset(sync_file)
        self._keys = get_keys(self._dataset)
        self.long_stim_threshold = long_stim_threshold
        if dff_file is not None:
            self.ophys_data_length = get_ophys_data_length(dff_file)
        else:
            self.ophys_data_length = None
        if stimulus_pkl is not None:
            self.stim_data_length = get_stim_data_length(stimulus_pkl)
        else:
            self.stim_data_length = None
        if eye_video is not None:
            self.eye_data_length = get_video_length(eye_video)
        else:
            self.eye_data_length = None
        if behavior_video is not None:
            self.behavior_data_length = get_video_length(behavior_video)
        else:
            self.behavior_data_length = None

    @property
    def dataset(self):
        return self._dataset


    @property
    def ophys_timestamps(self):
        """Get the timestamps for the ophys data."""
        ophys_key = self._keys["2p"]
        if self.scanner == "SCIVIVO":
            # Scientifica data looks different than Nikon.
            # http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
            times = self.dataset.get_rising_edges(ophys_key, units="seconds")
        elif self.scanner == "NIKONA1RMP":
            # Nikon has a signal that indicates when it started writing to disk
            acquiring_key = self._keys["acquiring"]
            acquisition_start = self._dataset.get_rising_edges(
                acquiring_key, units="seconds")[0]
            ophys_times = self._dataset.get_falling_edges(
                ophys_key, units="seconds")
            times = ophys_times[ophys_times >= acquisition_start]
        else:
            raise ValueError("Invalid scanner: {}".format(self.scanner))

        return times

    @property
    def corrected_ophys_timestamps(self):
        times = self.ophys_timestamps

        delta = 0
        if self.ophys_data_length is not None:
            if len(times) < self.ophys_data_length:
                raise ValueError(
                    "Got too few timestamps ({}) for ophys data length "
                    "({})".format(len(times), self.ophys_data_length))
            elif len(times) > self.ophys_data_length:
                logging.info("Ophys data of length %s has timestamps of "
                             "length %s, truncating timestamps",
                             self.ophys_data_length, len(times))
                delta = len(times) - self.ophys_data_length
                times = times[:-delta]
        else:
            logging.info("No data length provided for ophys stream")

        return times, delta

    @property
    def stim_timestamps(self):
        stim_key = self._keys["stimulus"]

        return self.dataset.get_falling_edges(stim_key, units="seconds")

    @property
    def corrected_stim_timestamps(self):
        timestamps = self.stim_timestamps

        delta = 0
        if self.stim_data_length is not None and \
           self.stim_data_length < len(timestamps):
            stim_key = self._keys["stimulus"]
            rising = self.dataset.get_rising_edges(stim_key, units="seconds")

            # Some versions of camstim caused a spike when the DAQ is first
            # initialized. Remove it.
            if rising[1] - rising[0] > self.long_stim_threshold:
                logging.info("Initial DAQ spike detected from stimulus, "
                             "removing it")
                timestamps = timestamps[1:]

            delta = len(timestamps) - self.stim_data_length
            if delta != 0:
                logging.info("Stim data of length %s has timestamps of "
                             "length %s",
                             self.stim_data_length, len(timestamps))
        elif self.stim_data_length is None:
            logging.info("No data length provided for stim stream")

        photodiode_key = self._keys["photodiode"]
        delay = monitor_delay(self.dataset, timestamps, photodiode_key)
        
        return timestamps + delay, delta

    @property
    def behavior_video_timestamps(self):
        key = self._keys["behavior_camera"]

        return self.dataset.get_falling_edges(key, units="seconds")

    @property
    def corrected_behavior_video_timestamps(self):
        return corrected_video_timestamps("Behavior video",
                                          self.behavior_video_timestamps,
                                          self.behavior_data_length)

    @property
    def eye_video_timestamps(self):
        key = self._keys["eye_camera"]

        return self.dataset.get_falling_edges(key, units="seconds")

    @property
    def corrected_eye_video_timestamps(self):
        return corrected_video_timestamps("Eye video",
                                          self.eye_video_timestamps,
                                          self.eye_data_length)
