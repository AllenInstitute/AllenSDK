import pytest
import numpy as np
from allensdk.internal.brain_observatory import eye_calibration

def cr_params():
    x, y = np.meshgrid(np.array([300, 320, 340]),
                       np.array([220, 240, 260]))
    return np.vstack((x.flatten(), y.flatten())).T

def pupil_params():
    x, y = np.meshgrid(np.array([280, 320, 380]),
                       np.array([200, 240, 280]))
    return np.vstack((x.flatten(), y.flatten())).T


@pytest.mark.parametrize("led_position,eye_radius", [
    (np.array([25.89, -6.12, 3.21]), 0.1682),
    (np.array([24.6, 9.23, 5.26]), 0.1682),
    (np.array([20.0, 20.0, 20.0]), 500),
])
def test_cr_position_in_mouse_eye_coordinates(led_position, eye_radius):
    cr = eye_calibration.EyeCalibration.cr_position_in_mouse_eye_coordinates(
        led_position, eye_radius)
    tol = 0.000000001
    assert(np.abs(np.linalg.norm(cr) - 0.5*eye_radius) < tol)
    err = np.abs(cr/np.linalg.norm(cr) - \
                 led_position/np.linalg.norm(led_position))
    assert(np.all(err < tol))


@pytest.mark.parametrize("led_position,camera_rotations", [
    (np.array([10.0, 0.0, 0.0]),np.array([0.0, 0.0, 0.0])),
    (np.array([10.0, 0.0, 0.0]),np.array([0.0, 0.0, np.pi/4]))
])
def test_pupil_position_in_mouse_eye_coordinates_right(
    led_position, camera_rotations):
    CM_PER_PIXEL = 10.2/10000
    TOL = 0.0000000001
    c = eye_calibration.EyeCalibration(
        led_position=led_position, cm_per_pixel=CM_PER_PIXEL,
        eye_radius=0.1682, camera_rotations=camera_rotations,
        camera_position=np.array([13.0, 0.0, 0.0]))
    pupil = pupil_params()
    cr = cr_params()
    bad = c.pupil_position_in_mouse_eye_coordinates(np.array([[1000, 0], [0, 1000]]),
                                                    np.array([[0, 0], [0, 0]]))
    assert(np.all(np.isnan(bad)))
    pos = c.pupil_position_in_mouse_eye_coordinates(pupil, cr)
    x = (pupil.T[0] - cr.T[0])*CM_PER_PIXEL
    y = (cr.T[1] - pupil.T[1])*CM_PER_PIXEL
    xr = x*np.cos(-camera_rotations[2]) - y*np.sin(-camera_rotations[2])
    yr = x*np.sin(-camera_rotations[2]) + y*np.cos(-camera_rotations[2])
    assert(np.all(np.abs(pos.T[2] - (yr + c.cr[2])) < TOL))
    assert(np.all(np.abs(pos.T[1] - (xr + c.cr[1])) < TOL))


@pytest.mark.parametrize("led_position,camera_rotations", [
    (np.array([10.0, 0.0, 0.0]),np.array([0.0, 0.0, 0.0])),
    (np.array([10.0, 0.0, 0.0]),np.array([0.0, 0.0, np.pi/4]))
])
def test_pupil_position_in_mouse_eye_coordinates_front(
    led_position, camera_rotations):
    CM_PER_PIXEL = 10.2/10000
    TOL = 0.0000000001
    c = eye_calibration.EyeCalibration(
        led_position=led_position, cm_per_pixel=CM_PER_PIXEL,
        eye_radius=0.1682, camera_rotations=camera_rotations,
        camera_position=np.array([0.0, 13.0, 0.0]))
    pupil = pupil_params()
    cr = cr_params()
    bad = c.pupil_position_in_mouse_eye_coordinates(np.array([[1000, 0], [0, 1000]]),
                                                    np.array([[0, 0], [0, 0]]))
    assert(np.all(np.isnan(bad)))
    pos = c.pupil_position_in_mouse_eye_coordinates(pupil, cr)
    x = (pupil.T[0] - cr.T[0])*CM_PER_PIXEL
    y = (cr.T[1] - pupil.T[1])*CM_PER_PIXEL
    xr = x*np.cos(-camera_rotations[2]) - y*np.sin(-camera_rotations[2])
    yr = x*np.sin(-camera_rotations[2]) + y*np.cos(-camera_rotations[2])
    assert(np.all(np.abs(pos.T[2] - (yr + c.cr[2])) < TOL))
    assert(np.all(np.abs(pos.T[0] + (xr - c.cr[0])) < TOL))
