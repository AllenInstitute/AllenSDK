import os
from allensdk.internal.core.lims_pipeline_module import (PipelineModule,
                                                         run_module)
from allensdk.internal.brain_observatory import (eye_calibration,
                                                 itracker_utils)
import allensdk.internal.core.lims_utilities as lu
from allensdk.internal.pipeline_modules.run_ophys_eye_calibration import write_output
import numpy as np
import h5py
import pandas as pd

EYE_RADIUS = 0.1682
CM_PER_PIXEL = 10.2/10000

def parse_input_data(data):
    dlc_df = pd.read_hdf(data["dcl_eye_flat_file"])

    pupil_params = np.empty((len(dlc_df), 5))
    pupil_params[:, 0] = dlc_df.pupil_ellipse_center_x.real
    pupil_params[:, 1] = dlc_df.pupil_ellipse_center_y.real
    pupil_params[:, 2] = 2*dlc_df.pupil_ellipse_width.real
    pupil_params[:, 3] = 2*dlc_df.pupil_ellipse_height.real
    pupil_params[:, 4] = np.rad2deg(dlc_df.pupil_ellipse_phi.real)

    cr_params = np.empty((len(dlc_df), 5))
    cr_params[:, 0] = dlc_df.reflection_x.real
    cr_params[:, 1] = dlc_df.reflection_y.real
    cr_params[:, 2] = float('nan')
    cr_params[:, 3] = float('nan')
    cr_params[:, 4] = float('nan')

    frame_width = data["frame_width"]
    frame_height = data["frame_height"]
    output_file = data["output_file"]
    monitor_position = np.array([
        float(data['monitor_position_x_mm'])/10.0,
        float(data['monitor_position_y_mm'])/10.0,
        float(data['monitor_position_z_mm'])/10.0,
    ])
    monitor_rotations = np.array([
        float(data['monitor_rotation_x_deg'])*np.pi/180,
        float(data['monitor_rotation_y_deg'])*np.pi/180,
        float(data['monitor_rotation_z_deg'])*np.pi/180,
    ])
    camera_position = np.array([
        float(data['camera_position_x_mm'])/10.0,
        float(data['camera_position_y_mm'])/10.0,
        float(data['camera_position_z_mm'])/10.0,
    ])
    camera_rotations = np.array([
        float(data['camera_rotation_x_deg'])*np.pi/180,
        float(data['camera_rotation_y_deg'])*np.pi/180,
        float(data['camera_rotation_z_deg'])*np.pi/180,
    ])
    led_position = np.array([
        float(data['led_position_x_mm'])/10.0,
        float(data['led_position_y_mm'])/10.0,
        float(data['led_position_z_mm'])/10.0,
    ])
    calibrator = eye_calibration.EyeCalibration(
        monitor_position=monitor_position,
        monitor_rotations=monitor_rotations,
        led_position=led_position,
        camera_position=camera_position,
        camera_rotations=camera_rotations,
        eye_radius=EYE_RADIUS,
        cm_per_pixel=CM_PER_PIXEL)

    cr_params, _ = itracker_utils.post_process_cr(cr_params)
    pupil_params = itracker_utils.post_process_pupil(pupil_params)
    cr_params = itracker_utils.filter_bad_params(cr_params, frame_width,
                                                 frame_height)
    pupil_params = itracker_utils.filter_bad_params(pupil_params, frame_width,
                                                    frame_height)
    return calibrator, cr_params, pupil_params, output_file



def main():
    mod = PipelineModule()
    data = mod.input_data()
    calibrator, cr_params, pupil_params, outfile = parse_input_data(data)

    pupil_areas = calibrator.compute_area(pupil_params)
    pupil_on_monitor_deg = calibrator.pupil_position_on_monitor_in_degrees(
        pupil_params, cr_params)
    pupil_on_monitor_cm = calibrator.pupil_position_on_monitor_in_cm(
        pupil_params, cr_params)
    missing_index = np.isnan(pupil_areas) | np.isnan(pupil_on_monitor_deg.T[0])
    pupil_areas[missing_index] = np.nan
    pupil_on_monitor_deg[missing_index, :] = np.nan
    pupil_on_monitor_cm[missing_index, :] = np.nan
    write_output(outfile, pupil_on_monitor_deg, pupil_on_monitor_cm,
                 pupil_areas)

    mod.write_output_data({"screen_mapping_file": outfile})

if __name__ == "__main__": main()

    