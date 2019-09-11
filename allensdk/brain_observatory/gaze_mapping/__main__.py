import h5py
import logging
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from argschema import ArgSchemaParser

from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs
)

from ._schemas import InputSchema, OutputSchema
from ._eye_calibration import compute_ellipse_areas, EyeCalibration
from ._filter_utils import (post_process_cr,
                            post_process_areas)
from ._sync_frames import get_synchronized_camera_frame_times


def repackage_input_args(parser_args: dict) -> dict:
    """Repackage arguments obtained by argschema.

    1) Converts individual coordinate/rotation fields to numpy
       position/rotation arrays.
    2) Converts path strings to pathlib Paths

    Args:
        parser_args (dict): Parsed args obtained from argschema.

    Returns:
        dict: Repackaged args.
    """
    new_args: dict = {}

    new_args["input_file"] = Path(parser_args["input_file"])
    new_args["session_sync_file"] = Path(parser_args["session_sync_file"])
    new_args["output_file"] = Path(parser_args["output_file"])

    monitor_position = np.array([parser_args["monitor_position_x_mm"],
                                 parser_args["monitor_position_y_mm"],
                                 parser_args["monitor_position_z_mm"]])
    new_args["monitor_position"] = monitor_position

    monitor_rotations_deg = np.array([parser_args["monitor_rotation_x_deg"],
                                      parser_args["monitor_rotation_y_deg"],
                                      parser_args["monitor_rotation_z_deg"]])
    new_args["monitor_rotations"] = np.radians(monitor_rotations_deg)

    camera_position = np.array([parser_args["camera_position_x_mm"],
                                parser_args["camera_position_y_mm"],
                                parser_args["camera_position_z_mm"]])
    new_args["camera_position"] = camera_position

    camera_rotations_deg = np.array([parser_args["camera_rotation_x_deg"],
                                     parser_args["camera_rotation_y_deg"],
                                     parser_args["camera_rotation_z_deg"]])
    new_args["camera_rotations"] = np.radians(camera_rotations_deg)

    led_position = np.array([parser_args["led_position_x_mm"],
                             parser_args["led_position_y_mm"],
                             parser_args["led_position_z_mm"]])
    new_args["led_position"] = led_position
    new_args["eye_radius_cm"] = parser_args["eye_radius_cm"]
    new_args["cm_per_pixel"] = parser_args["cm_per_pixel"]

    new_args["equipment"] = parser_args["equipment"]
    new_args["date_of_acquisition"] = parser_args["date_of_acquisition"]
    new_args["eye_video_file"] = Path(parser_args["eye_video_file"])
    return new_args


def run_gaze_mapping(pupil_parameters: pd.DataFrame,
                     cr_parameters: pd.DataFrame,
                     eye_parameters: pd.DataFrame,
                     monitor_position: np.ndarray,
                     monitor_rotations: np.ndarray,
                     camera_position: np.ndarray,
                     camera_rotations: np.ndarray,
                     led_position: np.ndarray,
                     eye_radius_cm: float,
                     cm_per_pixel: float) -> dict:
    """Map gaze positions onto monitor coordinates and
       calculate eye/pupil areas

    Note: Monitor and Camera positions/rotations are in their own coordinate
    systems which have are different from the eye coordinate system.

    Example: Z-axis for monitor and camera are aligned with X-axis for eye
    coordinate system

    Args:
        pupil_parameters (pd.DataFrame): A table of pupil parameters with
        5 columns ("center_x", "center_y", "height", "phi", "width")
        and n-row timepoints. Coordinate
        cr_parameters (pd.DataFrame): A table of corneal reflection params with
        5 columns ("center_x", "center_y", "height", "phi", "width")
        and n-row timepoints.
        eye_parameters (pd.DataFrame): A table of eye parameters with
        5 columns ("center_x", "center_y", "height", "phi", "width")
        and n-row timepoints.
        monitor_position (np.ndarray): An array describing monitor position
            [x, y, z]
        monitor_rotations (np.ndarray): An array describing monitor orientation
            about [x, y, z] axes.
        camera_position (np.ndarray): An array describing camera position
            [x, y, z]
        camera_rotations (np.ndarray): An array describing camera orientation
            about [x, y, z] axes.
        led_position (np.ndarray): An array describing LED position [x, y, z]
        eye_radius_cm (float): Radius of eye being tracked in cm.
        cm_per_pixel (float): Ratio of centimeters per pixel

    Returns:
        dict: A dictionary of gaze mapping outputs with
            fields for: `pupil_areas`, `eye_areas`, `pupil_on_monitor_cm`, and
            `pupil_on_monitor_deg`.
    """
    output = {}

    gaze_mapper = EyeCalibration(monitor_position=monitor_position,
                                 monitor_rotations=monitor_rotations,
                                 led_position=led_position,
                                 camera_position=camera_position,
                                 camera_rotations=camera_rotations,
                                 eye_radius=eye_radius_cm,
                                 cm_per_pixel=cm_per_pixel)

    raw_pupil_areas = compute_ellipse_areas(pupil_parameters)

    raw_eye_areas = compute_ellipse_areas(eye_parameters)

    # TODO 1: This previous implementation of gaze mapping takes in x, y coordinates
    # of pupil and CR centers but appears to be spitting out monitor locations
    # in y, x... This needs to be investigated in more detail...

    raw_pupil_on_monitor_cm = gaze_mapper.pupil_position_on_monitor_in_cm(
        pupil_parameters[["center_x", "center_y"]].values,
        cr_parameters[["center_x", "center_y"]].values
    )

    raw_pupil_on_monitor_deg = gaze_mapper.pupil_position_on_monitor_in_degrees(
        pupil_parameters[["center_x", "center_y"]].values,
        cr_parameters[["center_x", "center_y"]].values
    )

    # Make bool mask for all time indices where
    # pupil_area or eye_area or pupil_on_monitor_* is np.nan
    raw_nan_mask = (raw_pupil_areas.isna()
                    | raw_eye_areas.isna()
                    | np.isnan(raw_pupil_on_monitor_deg.T[0]))
    raw_pupil_areas[raw_nan_mask] = np.nan
    raw_eye_areas[raw_nan_mask] = np.nan
    raw_pupil_on_monitor_cm[raw_nan_mask, :] = np.nan
    raw_pupil_on_monitor_deg[raw_nan_mask, :] = np.nan

    output["raw_pupil_areas"] = pd.Series(raw_pupil_areas)
    output["raw_eye_areas"] = pd.Series(raw_eye_areas)
    # Swapping x and y columns (see TODO 1)
    output["raw_pupil_on_monitor_cm"] = pd.DataFrame(raw_pupil_on_monitor_cm, columns=["y_pos_cm", "x_pos_cm"])
    output["raw_pupil_on_monitor_deg"] = pd.DataFrame(raw_pupil_on_monitor_deg, columns=["y_pos_deg", "x_pos_deg"])

    # Perform post processing of data
    new_pupil_areas = raw_pupil_areas.copy()
    new_eye_areas = raw_eye_areas.copy()
    new_pupil_on_monitor_cm = raw_pupil_on_monitor_cm.copy()
    new_pupil_on_monitor_deg = raw_pupil_on_monitor_deg.copy()

    new_pupil_areas = post_process_areas(new_pupil_areas.values)
    new_eye_areas = post_process_areas(new_eye_areas.values)
    _, filtered_pos_indices = post_process_cr(cr_parameters[["center_x",
                                                             "center_y",
                                                             "phi",
                                                             "width",
                                                             "height"]].values)

    new_nan_mask = (np.isnan(new_pupil_areas)
                    | np.isnan(new_eye_areas)
                    | filtered_pos_indices)
    new_pupil_areas[new_nan_mask] = np.nan
    new_eye_areas[new_nan_mask] = np.nan
    new_pupil_on_monitor_cm[new_nan_mask, :] = np.nan
    new_pupil_on_monitor_deg[new_nan_mask, :] = np.nan

    output["new_pupil_areas"] = pd.Series(new_pupil_areas)
    output["new_eye_areas"] = pd.Series(new_eye_areas)
    # Swapping x and y columns (see TODO 1)
    output["new_pupil_on_monitor_cm"] = pd.DataFrame(new_pupil_on_monitor_cm, columns=["y_pos_cm", "x_pos_cm"])
    output["new_pupil_on_monitor_deg"] = pd.DataFrame(new_pupil_on_monitor_deg, columns=["y_pos_deg", "x_pos_deg"])

    return output


def write_gaze_mapping_output_to_h5(output_savepath: Path,
                                    gaze_map_output: dict):
    """Write output of gaze mapping to an h5 file.

    Args:
        output_savepath (Path): Desired output save path
        gaze_map_output (dict): A dictionary of gaze mapping outputs with
            fields for: `pupil_areas`, `eye_areas`, `pupil_on_monitor_cm`, and
            `pupil_on_monitor_deg`.
    """

    gaze_map_output["raw_eye_areas"].to_hdf(output_savepath, key="raw_eye_areas", mode="w")
    gaze_map_output["raw_pupil_areas"].to_hdf(output_savepath, key="raw_pupil_areas", mode="a")
    gaze_map_output["raw_pupil_on_monitor_cm"].to_hdf(output_savepath, key="raw_screen_coordinates", mode="a")
    gaze_map_output["raw_pupil_on_monitor_deg"].to_hdf(output_savepath, key="raw_screen_coordinates_spherical", mode="a")

    gaze_map_output["new_eye_areas"].to_hdf(output_savepath, key="new_eye_areas", mode="a")
    gaze_map_output["new_pupil_areas"].to_hdf(output_savepath, key="new_pupil_areas", mode="a")
    gaze_map_output["new_pupil_on_monitor_cm"].to_hdf(output_savepath, key="new_screen_coordinates", mode="a")
    gaze_map_output["new_pupil_on_monitor_deg"].to_hdf(output_savepath, key="new_screen_coordinates_spherical", mode="a")

    gaze_map_output["synced_frame_timestamps_sec"].to_hdf(output_savepath, key="synced_frame_timestamps", mode="a")


def main():

    logging.basicConfig(format=('%(asctime)s:%(funcName)s'
                                ':%(levelname)s:%(message)s'))

    parser = ArgSchemaParser(args=sys.argv[1:],
                             schema_type=InputSchema,
                             output_schema_type=OutputSchema)

    args = repackage_input_args(parser.args)

    # TODO 2: Some ellipses.h5 files have the 'cr' key as complex type instead of
    # float. For now, when loading ellipses.h5 files, always coerce to float
    # but this should eventually be resolved upstream...
    pupil_params = pd.read_hdf(args['input_file'], key="pupil").astype(float)
    cr_params = pd.read_hdf(args['input_file'], key="cr").astype(float)
    eye_params = pd.read_hdf(args['input_file'], key="eye").astype(float)

    num_frames_match = ((pupil_params.shape[0] == cr_params.shape[0]) and
                        (cr_params.shape[0] == eye_params.shape[0]))
    if not num_frames_match:
        raise RuntimeError("The number of frames for ellipse fits don't "
                           "match when they should: "
                           f"pupil_params ({pupil_params.shape[0]}), "
                           f"cr_params ({cr_params.shape[0]}), "
                           f"eye_params ({eye_params.shape[0]}).")

    output = run_gaze_mapping(pupil_parameters=pupil_params,
                              cr_parameters=cr_params,
                              eye_parameters=eye_params,
                              monitor_position=args["monitor_position"],
                              monitor_rotations=args["monitor_rotations"],
                              camera_position=args["camera_position"],
                              camera_rotations=args["camera_rotations"],
                              led_position=args["led_position"],
                              eye_radius_cm=args["eye_radius_cm"],
                              cm_per_pixel=args["cm_per_pixel"])

    # Add synchronized frame times
    frame_times = get_synchronized_camera_frame_times(args["session_sync_file"])
    if (pupil_params.shape[0] != len(frame_times)):
        raise RuntimeError("The number of camera sync pulses in the "
                           f"sync file ({len(frame_times)}) do not match "
                           "with the number of eye tracking frames "
                           f"({pupil_params.shape[0]})!!!")
    output["synced_frame_timestamps_sec"] = frame_times

    write_gaze_mapping_output_to_h5(args["output_file"], output)
    module_output = {"screen_mapping_file": str(args["output_file"])}
    write_or_print_outputs(module_output, parser)


if __name__ == "__main__":
    main()
