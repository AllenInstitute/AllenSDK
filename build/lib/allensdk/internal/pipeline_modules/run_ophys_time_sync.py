import logging
import argparse
from allensdk.internal.core.lims_pipeline_module import PipelineModule
import allensdk.core.json_utilities as ju
from allensdk.internal.brain_observatory import time_sync as ts
import numpy as np
import h5py
import os


def write_output(output_file, ophys_times, stim_alignment, eye_alignment,
                 behavior_alignment, ophys_delta, stim_delta, eye_delta,
                 behavior_delta):
    with h5py.File(output_file, "w") as f:
        f['stimulus_alignment'] = stim_alignment
        f['eye_tracking_alignment'] = eye_alignment
        f['body_camera_alignment'] = behavior_alignment
        f['twop_vsync_fall'] = ophys_times
        f['ophys_delta'] = ophys_delta
        f['stim_delta'] = stim_delta
        f['eye_delta'] = eye_delta
        f['behavior_delta'] = behavior_delta


def main():
    parser = argparse.ArgumentParser("Generate brain observatory alignment.")
    parser.add_argument('input_json')
    parser.add_argument('--log-level', default=logging.DEBUG)
    mod = PipelineModule("Generate brain observatory alignment.", parser)

    input_data = mod.input_data()
    experiment_id = input_data.pop("ophys_experiment_id")
    sync_file = input_data.pop("sync_file")
    output_file = input_data.pop("output_file")

    aligner = ts.OphysTimeAligner(sync_file, **input_data)

    ophys_times, ophys_delta = aligner.corrected_ophys_timestamps
    stim_times, stim_delta = aligner.corrected_stim_timestamps
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

    write_output(output_file, ophys_times, stim_alignment, eye_alignment,
                 behavior_alignment, ophys_delta, stim_delta, eye_delta,
                 beh_delta)


if __name__ == "__main__": main()