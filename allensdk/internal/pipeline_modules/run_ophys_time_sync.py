import logging
import argparse
import os
import datetime
import json

import numpy as np
import h5py

import allensdk
from allensdk.internal.core.lims_pipeline_module import PipelineModule
from allensdk.internal.brain_observatory import time_sync as ts
from allensdk.brain_observatory.argschema_utilities import \
    check_write_access_overwrite


def write_output_h5(
    output_file, ophys_times, stim_alignment, eye_alignment, 
    behavior_alignment, ophys_delta, stim_delta, stim_delay, eye_delta, 
    behavior_delta
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f['stimulus_alignment'] = stim_alignment
        f['eye_tracking_alignment'] = eye_alignment
        f['body_camera_alignment'] = behavior_alignment
        f['twop_vsync_fall'] = ophys_times
        f['ophys_delta'] = ophys_delta
        f['stim_delta'] = stim_delta
        f['stim_delay'] = stim_delay
        f['eye_delta'] = eye_delta
        f['behavior_delta'] = behavior_delta


def write_output_json(
    path, ophys_delta, stim_delta, stim_delay, eye_delta, behavior_delta
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as output_json:
        json.dump({
            "allensdk_version": allensdk.__version__,
            "date": str(datetime.datetime.now()),
            "ophys_delta": ophys_delta,
            "stim_delta": stim_delta,
            "stim_delay": stim_delay,
            "eye_delta": eye_delta,
            "behavior_delta": behavior_delta
        }, output_json, indent=2)


def write_outputs(
    output_h5_path, output_json_path, 
    ophys_times, stim_alignment, eye_alignment, behavior_alignment, 
    ophys_delta, stim_delta, stim_delay, eye_delta, behavior_delta
):
    """
    """

    if output_json_path is not None:
        write_output_json(
            output_json_path, ophys_delta, stim_delta, stim_delay, eye_delta, 
            behavior_delta
        )

    write_output_h5(
        output_h5_path, ophys_times, stim_alignment, eye_alignment,
        behavior_alignment, ophys_delta, stim_delta, stim_delay, eye_delta,
        behavior_delta
    )


def check_outputs_writable(output_json_path, output_h5_path):
    """ Make sure we can actually write to the specified output paths, 
    preferably before running an expensive calculation. Allows for creation
    of intermediate directories
    """

    check_write_access_overwrite(output_h5_path)

    if output_json_path is not None:
        check_write_access_overwrite(output_json_path)


def run_ophys_time_sync(input_data, experiment_id, sync_file):

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

    return (ophys_times, ophys_delta, stim_times, stim_delta, stim_delay, 
        eye_times, eye_delta, beh_times, beh_delta, stim_alignment, 
        eye_alignment, behavior_alignment)


def main():
    parser = argparse.ArgumentParser("Generate brain observatory alignment.")
    parser.add_argument('input_json', type=str, 
        help="path to input json")
    parser.add_argument("output_json", type=str, nargs="?", 
        help="path to which output json will be written")
    parser.add_argument('--log-level', default=logging.DEBUG)
    mod = PipelineModule("Generate brain observatory alignment.", parser)

    input_data = mod.input_data()
    experiment_id = input_data.pop("ophys_experiment_id")
    sync_file = input_data.pop("sync_file")

    output_file = input_data.pop("output_file")
    output_json_path = mod.args.output_json
    check_outputs_writable(output_json_path, output_file)

    (
        ophys_times, ophys_delta, stim_times, stim_delta, stim_delay, 
        eye_times, eye_delta, beh_times, beh_delta, stim_alignment, 
        eye_alignment, behavior_alignment
    ) = run_ophys_time_sync(input_data, experiment_id, sync_file)

    write_outputs(
        output_file, output_json_path, 
        ophys_times, stim_alignment, eye_alignment, behavior_alignment, 
        ophys_delta, stim_delta, stim_delay, eye_delta, beh_delta
    )



if __name__ == "__main__": main()