import logging
import argparse
import os
import datetime
import json
from typing import NamedTuple, Optional, Dict

import numpy as np
import h5py

import allensdk
from allensdk.internal.core.lims_pipeline_module import PipelineModule
from allensdk.internal.brain_observatory import time_sync as ts
from allensdk.brain_observatory.argschema_utilities import \
    check_write_access_overwrite


class TimeSyncOutputs(NamedTuple):
    """ Schema for synchronization outputs
    """

    # unique identifier for the experiment being aligned
    experiment_id: int

    # calculated monitor delay (s)
    stimulus_delay: float

    # For each data stream, the count of "extra" timestamps (compared to the 
    # number of samples)
    ophys_delta: int
    stimulus_delta: int
    eye_delta: int
    behavior_delta: int

    # aligned timestamps for each data stream (s)
    ophys_times: np.ndarray 
    stimulus_times: np.ndarray
    eye_times: np.ndarray
    behavior_times: np.ndarray

    # for non-ophys data streams, a mapping from samples to corresponding ophys 
    # frames
    stimulus_alignment: np.ndarray
    eye_alignment: np.ndarray
    behavior_alignment: np.ndarray


class TimeSyncWriter:

    def __init__(
        self, 
        output_h5_path: str, 
        output_json_path: Optional[str] = None
    ):
        """ Writes synchronization outputs to h5 and (optionally) json.

        Parameters
        ----------
        output_h5_path : "heavy" outputs (e.g aligned timestamps and 
            ophy frame correspondances) will ONLY be stored here. Lightweight 
            outputs (e.g. stimulus delay) will also be written here as scalars.
        output_json_path : if provided, lightweight outputs will be written 
            here, along with provenance information, such as the date and 
            allensdk version.

        """

        self.output_h5_path: str = output_h5_path
        self.output_json_path: Optional[str] = output_json_path

    def validate_paths(self):
        """ Determines whether we can actually write to the specified paths, 
        allowing for creation of intermediate directories. It is a good idea 
        to run this beore doing any heavy calculations!
        """

        check_write_access_overwrite(self.output_h5_path)

        if self.output_json_path is not None:
            check_write_access_overwrite(self.output_json_path)

    def write(self, outputs: TimeSyncOutputs):
        """ Convenience for writing both an output h5 and (if applicable) an 
        output json.

        Parameters
        ----------
        outputs : the data to be written

        """

        self.write_output_h5(outputs)

        if self.output_json_path is not None:
            self.write_output_json(outputs)

    def write_output_h5(self, outputs):
        """ Write (mainly) heaviweight data to an h5 file.

        Parameters
        ----------
        outputs : the data to be written

        """

        os.makedirs(os.path.dirname(self.output_h5_path), exist_ok=True)

        with h5py.File(self.output_h5_path, "w") as output_h5:
            output_h5["stimulus_alignment"] = outputs.stimulus_alignment
            output_h5["eye_tracking_alignment"] = outputs.eye_alignment
            output_h5["body_camera_alignment"] = outputs.behavior_alignment
            output_h5["twop_vsync_fall"] = outputs.ophys_times
            output_h5["ophys_delta"] = outputs.ophys_delta
            output_h5["stim_delta"] = outputs.stimulus_delta
            output_h5["stim_delay"] = outputs.stimulus_delay
            output_h5["eye_delta"] = outputs.eye_delta
            output_h5["behavior_delta"] = outputs.behavior_delta

    def write_output_json(self, outputs):
        """ Write lightweight data to a json

        Parameters
        ----------
        outputs : the data to be written

        """
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)

        with open(self.output_json_path, "w") as output_json:
            json.dump({
                "allensdk_version": allensdk.__version__,
                "date": str(datetime.datetime.now()),
                "experiment_id": outputs.experiment_id,
                "output_h5_path": self.output_h5_path,
                "ophys_delta": outputs.ophys_delta,
                "stim_delta": outputs.stimulus_delta,
                "stim_delay": outputs.stimulus_delay,
                "eye_delta": outputs.eye_delta,
                "behavior_delta": outputs.behavior_delta
            }, output_json, indent=2)


def run_ophys_time_sync(
    input_data: Dict, 
    experiment_id: int,
    sync_file: str
) -> TimeSyncOutputs:
    """ Carry out synchronization of timestamps across the data streams of an 
    ophys experiment.

    Parameters
    ----------
    input_data : parameters that will be provided to the OphysTimeAligner 
        driving this synchronization.
    experiment_id: unique identifier for the experiment being aligned
    sync_file : path to h5 file containing sync timestamps

    Returns
    -------
    A TimeSyncOutputs (see definintion for more information) of output 
        parameters and arrays of aligned timestamps.

    """

    aligner = ts.OphysTimeAligner(sync_file, **input_data)

    ophys_times, ophys_delta = aligner.corrected_ophys_timestamps
    stim_times, stim_delta, stim_delay = aligner.corrected_stim_timestamps
    eye_times, eye_delta = aligner.corrected_eye_video_timestamps
    beh_times, beh_delta = aligner.corrected_behavior_video_timestamps

    # stim array is index of ophys frame for each stim frame to match to
    # so len(stim_times)
    stim_alignment = ts.get_alignment_array(ophys_times, stim_times)

    # camera arrays are index of camera frame for each ophys frame ...
    # cam_nwb_creator depends on this so keeping it that way even though
    # it makes little sense... len(video_times)
    eye_alignment = ts.get_alignment_array(eye_times, ophys_times,
                                           int_method=np.ceil)

    behavior_alignment = ts.get_alignment_array(beh_times, ophys_times,
                                                int_method=np.ceil)

    return TimeSyncOutputs(
        experiment_id,
        stim_delay,
        ophys_delta,
        stim_delta,
        eye_delta,
        beh_delta,
        ophys_times,
        stim_times,
        eye_times,
        beh_times,
        stim_alignment,
        eye_alignment,
        behavior_alignment
    )


def main():
    parser = argparse.ArgumentParser("Generate brain observatory alignment.")
    parser.add_argument("input_json", type=str, 
        help="path to input json")
    parser.add_argument("output_json", type=str, nargs="?",
        help="path to which output json will be written")
    parser.add_argument("--log-level", default=logging.DEBUG)
    mod = PipelineModule("Generate brain observatory alignment.", parser)

    input_data = mod.input_data()
    experiment_id = input_data.pop("ophys_experiment_id")
    sync_file = input_data.pop("sync_file")

    writer = TimeSyncWriter(input_data.pop("output_file"), mod.args.output_json)
    writer.validate_paths()

    outputs = run_ophys_time_sync(input_data, experiment_id, sync_file)
    writer.write(outputs)


if __name__ == "__main__": main()