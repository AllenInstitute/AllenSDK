import pytest

import numpy as np
import pandas as pd

from allensdk.brain_observatory.gaze_mapping import _gaze_mapper as gm


@pytest.fixture()
def gaze_mapper_fixture(request):
    default_params = {
        "monitor_position": np.array([0, 0, 0]),
        "monitor_rotations": np.array([0, 0, 0]),
        "led_position": np.array([0, 0, 0]),
        "camera_position": np.array([0, 0, 0]),
        "camera_rotations": np.array([0, 0, 0]),
        "eye_radius": 0.1682,
        "cm_per_pixel": (10.2 / 10000.0)
    }
    default_params.update(request.param)
    return gm.GazeMapper(**default_params)


@pytest.fixture()
def rig_component_fixture(request):
    default_params = {
        "position_in_eye_coord_frame": np.array([0, 0, 0]),
        "rotations_in_self_coord_frame": np.array([0, 0, 0])
    }
    default_params.update(request.param)
    return gm.EyeTrackingRigObject(**default_params)


# ======== EyeTrackingRigObject tests ========
@pytest.mark.parametrize('rig_component_fixture,expected', [
    ({"position_in_eye_coord_frame": [1, 0, 0]},
     [[0, 0, -1],
      [-1, 0, 0],
      [0, 1, 0]]),

    ({"position_in_eye_coord_frame": [0, 1, 0]},
     [[1, 0, 0],
      [0, 0, -1],
      [0, 1, 0]]),

    ({"position_in_eye_coord_frame": [0, 0, 1]},
     [[0, -1, 0],
      [-1, 0, 0],
      [0, 0, -1]]),
], indirect=['rig_component_fixture'])
def test_generate_self_to_eye_frame_xform(rig_component_fixture, expected):
    obtained = rig_component_fixture.generate_self_to_eye_frame_xform()
    assert np.allclose(obtained.as_dcm(), expected)


# ======== GazeMapper tests ========
@pytest.mark.parametrize('gaze_mapper_fixture,expected', [
    # Simple 2D scenarios
    ({"led_position": np.array([100, 0, 50])}, np.array([0.08417078, 0, 0.04208539])),
    ({"led_position": np.array([50, 0, 20])}, np.array([0.08424169, 0, 0.03369668])),
    # 3D scenarios
    ({"led_position": np.array([246, 92.3, 52.6])}, np.array([0.07876523, 0.02955297, 0.01684167])),
    ({"led_position": np.array([258.9, -61.2, 32.1])}, np.array([0.08187032, -0.01935289, 0.01015078]))

], indirect=["gaze_mapper_fixture"])
def test_compute_cr_coordinate(gaze_mapper_fixture, expected):
    obtained = gaze_mapper_fixture.compute_cr_coordinate()
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize("gaze_mapper_fixture, method_inputs, expected", [
    ({"monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([150, 0, 0]),
      "led_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[300, 300], [350, 350], [325, 325], [290, 290]]),
      "cam_cr_params": np.array([[300, 300], [300, 300], [300, 300], [300, 300]])},
     [[-0.16820, 0.0000, 0.0000],
      [-0.15195, -0.0510, 0.0510],
      [-0.16428, -0.0255, 0.0255],
      [-0.16758, 0.0102, -0.0102]]),

    # Test when params result in estimated pupil location outside of eye
    ({"monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([150, 0, 0]),
      "led_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[900, 900], [350, 350], [325, 325], [250, 250], [100, 100]]),
      "cam_cr_params": np.array([[300, 300], [300, 300], [300, 300], [300, 300], [800, 800]])},
     [[np.nan, np.nan, np.nan],
      [-0.15195, -0.0510, 0.0510],
      [-0.16428, -0.0255, 0.0255],
      [-0.15195, 0.051, -0.051],
      [np.nan, np.nan, np.nan]]),

], indirect=['gaze_mapper_fixture'])
def test_pupil_pos_in_eye_coords(gaze_mapper_fixture,
                                 method_inputs,
                                 expected):
    obtained = gaze_mapper_fixture.pupil_pos_in_eye_coords(**method_inputs)
    assert np.allclose(obtained, expected, rtol=1e-4, equal_nan=True)


@pytest.mark.parametrize("gaze_mapper_fixture, method_inputs, expected", [
    ({"monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([150, 0, 0]),
      "led_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[300, 300], [350, 350], [325, 325], [290, 290]]),
      "cam_cr_params": np.array([[300, 300], [300, 300], [300, 300], [300, 300]])},
     [[0, 0],
      [-57.057, -57.057],
      [-26.386, -26.386],
      [10.3472, 10.347]]),

    # Test when params result in estimated pupil location outside of eye
    ({"monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([150, 0, 0]),
      "led_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[300, 300], [900, 900], [325, 325], [200, 200]]),
      "cam_cr_params": np.array([[300, 300], [300, 300], [300, 300], [800, 800]])},
     [[0, 0],
      [np.nan, np.nan],
      [-26.386, -26.386],
      [np.nan, np.nan]]),

], indirect=['gaze_mapper_fixture'])
def test_pupil_position_on_monitor_in_cm(gaze_mapper_fixture,
                                         method_inputs,
                                         expected):
    obtained = gaze_mapper_fixture.pupil_position_on_monitor_in_cm(**method_inputs)
    assert np.allclose(obtained, expected, rtol=1e-4, equal_nan=True)


@pytest.mark.parametrize("gaze_mapper_fixture, method_inputs, expected", [
    ({"monitor_position": np.array([170, 0, 0])},  # rig geometry parameters
     {"pupil_pos_on_monitor_in_cm": np.array([[2, 5]])},
     np.array([[0.6740368979845053, 1.6845678100189891]])),  # expected

    ({"monitor_position": np.array([100, 0, 0])},
     {"pupil_pos_on_monitor_in_cm": np.array([[5, 6], [8, 9]])},
     np.array([[2.862405226111748, 3.429356585864454],
               [4.573921259900861, 5.126473695179203]]))
], indirect=['gaze_mapper_fixture'])
def test_pupil_position_on_monitor_in_degrees(gaze_mapper_fixture,
                                              method_inputs,
                                              expected):
    obtained = gaze_mapper_fixture.pupil_position_on_monitor_in_degrees(
        **method_inputs
    )
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('gaze_mapper_fixture,ellipse_fits,expected,deg_diff_tolerance', [
    # General sanity check using extreme pupil values to see if output
    # screen mapped coordinates are generally in the right quadrant/hemisphere.

    # As if looking at top half of screen
    ({"led_position": np.array([135, 0, 0]),  # rig geometry parameters
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[300, 200]]),  # Pupil center (x, y) coords
      "cam_cr_params": np.array([[300, 300]])},  # Corneal reflect (x, y) coords
     np.array([[0, 1]]),  # Expected general direction of outputs in unit vector form
     0),  # Allowed angle tolerance (in degrees) between `expected - obtained`

    # As if looking at bottom half of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[300, 400]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[0, -1]]),
     0),

    # As if looking at right side of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[200, 300]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[1, 0]]),
     0),

    # As if looking at left side of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[400, 300]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[-1, 0]]),
     0),

    # As if looking at upper right quadrant of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[200, 200]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[1, 1]]),
     0),

    # As if looking at lower right quadrant of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[200, 400]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[1, -1]]),
     0),

    # As if looking to upper right quadrant of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[400, 200]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[-1, 1]]),
     0),

    # As if looking at lower left quadrant of screen
    ({"led_position": np.array([135, 0, 0]),
      "monitor_position": np.array([170, 0, 0]),
      "camera_position": np.array([130, 0, 0])},
     {"cam_pupil_params": np.array([[400, 400]]),
      "cam_cr_params": np.array([[300, 300]])},
     np.array([[-1, -1]]),
     0),

], indirect=["gaze_mapper_fixture"])
def test_mapping_gives_sane_outputs(gaze_mapper_fixture, ellipse_fits, expected, deg_diff_tolerance):
    obtained = gaze_mapper_fixture.pupil_position_on_monitor_in_cm(**ellipse_fits)
    for obt, exp in zip(obtained, expected):
        # Check that angle between the obtained monitor coordinate and expected
        # general direction is not more than the `deg_diff_tolerance`.
        obt_unit_vec = obt / np.linalg.norm(obt)
        angle_between = np.arccos(np.clip(np.dot(obt_unit_vec, exp), -1.0, 1.0))
        assert angle_between <= np.radians(deg_diff_tolerance)


# ======== Standalone function tests ========
@pytest.mark.parametrize('ellipse_params,expected', [
    (pd.DataFrame({"height": [1, 1, 1, 1], "width": [2, 2, 2, 2]}),
     pd.Series([4 * np.pi] * 4)),

    (pd.DataFrame({"height": [2, 2, 2, 2], "width": [1, 1, 1, 1]}),
     pd.Series([4 * np.pi] * 4)),

    (pd.DataFrame({"height": [2, 4, 8, 16], "width": [1, 3, 9, 27]}),
     pd.Series([4 * np.pi, 16 * np.pi, 81 * np.pi, 729 * np.pi])),

    (pd.DataFrame({"height": [1, 3, 9, 27], "width": [2, 4, 8, 16]}),
     pd.Series([4 * np.pi, 16 * np.pi, 81 * np.pi, 729 * np.pi])),

    (pd.DataFrame({"height": [np.nan, 3, np.nan, 27],
                  "width": [2, 4, np.nan, np.nan]}),
     pd.Series([4 * np.pi, 16 * np.pi, np.nan, 729 * np.pi])),
])
def test_compute_circular_areas(ellipse_params, expected):
    obtained = gm.compute_circular_areas(ellipse_params)

    assert np.allclose(obtained, expected, equal_nan=True)


@pytest.mark.parametrize('ellipse_params, expected', [
    (pd.DataFrame({"height": [1, 2, 3, 4], "width": [4, 3, 2, 1]}),
     pd.Series([4 * np.pi, 6 * np.pi, 6 * np.pi, 4 * np.pi])),

    (pd.DataFrame({"height": [np.nan, 7, 11, 12, np.nan],
                   "width": [5, 3, 11, np.nan, np.nan]}),
     pd.Series([np.nan, np.pi * 21, np.pi * 121, np.nan, np.nan]))
])
def test_compute_elliptical_areas(ellipse_params, expected):
    obtained = gm.compute_elliptical_areas(ellipse_params)

    assert np.allclose(obtained, expected, equal_nan=True)


@pytest.mark.parametrize("function_inputs,expected", [
    ({"plane_normal": np.array([1, 1, 1]),
      "plane_point": np.array([1, 1, -5]),
      "line_vectors": np.array([[6, 1, 4]]),
      "line_points": np.array([[-5, 1, -1]])},
     np.array([-3.90909091, 1.18181818, -0.27272727])),

    ({"plane_normal": np.array([1, 0, 0]),
      "plane_point": np.array([10, 0, 0]),
      "line_vectors": np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0]]),
      "line_points": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])},
     np.array([[10, 0, 0], [10, 10, 10], [10, 10, 0]])),

    ({"plane_normal": np.array([1, 0, 0]),
      "plane_point": np.array([10, 0, 0]),
      "line_vectors": np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]]),
      "line_points": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])},
     np.array([[10, 0, 0], [10, 10, 10], [10, 10, 0], [10, 0, 0], [10, 0, 0]])),

    ({"plane_normal": np.array([1, 1, 1]),
      "plane_point": np.array([10, 10, 10]),
      "line_vectors": np.array([[1, 1, 1], [1, 1, 1]]),
      "line_points": np.array([[1, 2, 3], [0, 0, 0]])},
     np.array([[9, 10, 11], [10, 10, 10]])),

    ({"plane_normal": np.array([2, 1, -4]),
      "plane_point": np.array([1, 1, -0.25]),
      "line_vectors": np.array([[1, 3, 1]]),
      "line_points": np.array([[0, 2, 0]])},
     np.array([[2, 8, 2]])),
])
def test_project_to_plane(function_inputs, expected):
    obtained = gm.project_to_plane(**function_inputs)
    print(obtained)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize("function_inputs, expected", [
    ({'x_rotation': 0.5,
      'y_rotation': 0.5,
      'z_rotation': 0.5},
     [[0.77015115, -0.21902415, 0.59907898],
      [0.42073549, 0.88034656, -0.21902415],
      [-0.47942554, 0.42073549, 0.77015115]]),

    ({'x_rotation': 0.5,
      'y_rotation': 0,
      'z_rotation': 0},
     [[1.0, 0.0, 0.0],
      [0.0, 0.87758256, -0.47942554],
      [0.0, 0.47942554, 0.87758256]]),

    ({'x_rotation': 0,
      'y_rotation': 0.5,
      'z_rotation': 0},
     [[0.87758256, 0.0, 0.47942554],
      [0.0, 1.0, 0.0],
      [-0.47942554, 0.0, 0.87758256]]),

    ({'x_rotation': 0,
      'y_rotation': 0,
      'z_rotation': 0.5},
     [[0.87758256, -0.47942554, 0.0],
      [0.47942554, 0.87758256, 0.0],
      [0.0, 0.0, 1.0]]),
])
def test_generate_object_rotation_xform(function_inputs, expected):
    obtained = gm.generate_object_rotation_xform(**function_inputs)
    assert np.allclose(obtained.as_dcm(), expected)
