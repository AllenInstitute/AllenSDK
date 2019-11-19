from pathlib import Path
import pytest

import numpy as np
import pandas as pd

import allensdk.brain_observatory.gaze_mapping.__main__ as main


def create_sample_ellipse_hdf(output_file: Path,
                              cr_data: pd.DataFrame,
                              eye_data: pd.DataFrame,
                              pupil_data: pd.DataFrame):
    cr_data.to_hdf(output_file, key='cr', mode='w')
    eye_data.to_hdf(output_file, key='eye', mode='a')
    pupil_data.to_hdf(output_file, key='pupil', mode='a')


@pytest.fixture
def ellipse_fits_fixture(tmp_path, request) -> dict:
    cr = {"center_x": [300, 305, 295, 310, 280],
          "center_y": [300, 305, 295, 310, 280],
          "width": [7, 8, 6, 7, 10],
          "height": [6, 9, 5, 6, 8],
          "phi": [0, 0.1, 0.15, 0.1, 0]}

    eye = {"center_x": [300, 305, 295, 310, 280],
           "center_y": [300, 305, 295, 310, 280],
           "width": [150, 155, 160, 150, 155],
           "height": [120, 115, 120, 110, 100],
           "phi": [0, 0.1, 0.15, 0.1, 0]}

    pupil = {"center_x": [300, 305, 295, 310, 280],
             "center_y": [300, 305, 295, 310, 280],
             "width": [30, 35, 40, 25, 50],
             "height": [25, 27, 30, 20, 45],
             "phi": [0, 0.1, 0.15, 0.1, 0]}

    test_dir = tmp_path / "test_load_ellipse_fit_params"
    test_dir.mkdir()

    if request.param["create_good_fits_file"]:
        test_path = test_dir / "good_ellipse_fits.h5"
    else:
        test_path = test_dir / "bad_ellipse_fits.h5"
        pupil = {"center_x": [300], "center_y": [300], "width": [30],
                 "height": [25], "phi": [0]}

    cr = pd.DataFrame(cr)
    eye = pd.DataFrame(eye)
    pupil = pd.DataFrame(pupil)

    create_sample_ellipse_hdf(test_path, cr, eye, pupil)

    return {"cr": pd.DataFrame(cr),
            "eye": pd.DataFrame(eye),
            "pupil": pd.DataFrame(pupil),
            "file_path": test_path}


@pytest.mark.parametrize("ellipse_fits_fixture, expect_good_file", [
    ({"create_good_fits_file": True}, True),
    ({"create_good_fits_file": False}, False)
], indirect=["ellipse_fits_fixture"])
def test_load_ellipse_fit_params(ellipse_fits_fixture: dict, expect_good_file: bool):
    expected = {"cr_params": pd.DataFrame(ellipse_fits_fixture["cr"]).astype(float),
                "pupil_params": pd.DataFrame(ellipse_fits_fixture["pupil"]).astype(float),
                "eye_params": pd.DataFrame(ellipse_fits_fixture["eye"]).astype(float)}

    if expect_good_file:
        obtained = main.load_ellipse_fit_params(ellipse_fits_fixture["file_path"])
        for key in expected.keys():
            pd.testing.assert_frame_equal(obtained[key], expected[key])
    else:
        with pytest.raises(RuntimeError, match="ellipse fits don't match"):
            obtained = main.load_ellipse_fit_params(ellipse_fits_fixture["file_path"])


@pytest.mark.parametrize("input_args, expected", [
    ({"input_file": Path("input_file.h5"),
      "session_sync_file": Path("sync_file.h5"),
      "output_file": Path("output_file.h5"),
      "monitor_position_x_mm": 100.0,
      "monitor_position_y_mm": 500.0,
      "monitor_position_z_mm": 300.0,
      "monitor_rotation_x_deg": 30,
      "monitor_rotation_y_deg": 60,
      "monitor_rotation_z_deg": 90,
      "camera_position_x_mm": 200.0,
      "camera_position_y_mm": 600.0,
      "camera_position_z_mm": 700.0,
      "camera_rotation_x_deg": 20,
      "camera_rotation_y_deg": 180,
      "camera_rotation_z_deg": 5,
      "led_position_x_mm": 800.0,
      "led_position_y_mm": 900.0,
      "led_position_z_mm": 1000.0,
      "eye_radius_cm": 0.1682,
      "cm_per_pixel": 0.0001,
      "equipment": "Rig A",
      "date_of_acquisition": "Some Date",
      "eye_video_file": Path("eye_video.avi")},

     {"pupil_params": "pupil_params_placeholder",
      "cr_params": "cr_params_placeholder",
      "eye_params": "eye_params_placeholder",
      "session_sync_file": Path("sync_file.h5"),
      "output_file": Path("output_file.h5"),
      "monitor_position": np.array([10.0, 50.0, 30.0]),
      "monitor_rotations": np.array([np.pi / 6, np.pi / 3, np.pi / 2]),
      "camera_position": np.array([20.0, 60.0, 70.0]),
      "camera_rotations": np.array([np.pi / 9, np.pi, np.pi / 36]),
      "led_position": np.array([80.0, 90.0, 100.0]),
      "eye_radius_cm": 0.1682,
      "cm_per_pixel": 0.0001,
      "equipment": "Rig A",
      "date_of_acquisition": "Some Date",
      "eye_video_file": Path("eye_video.avi")}
     ),

])
def test_preprocess_input_args(monkeypatch, input_args: dict, expected: dict):
    def mock_load_ellipse_fit_params(*args, **kwargs):
        return {"pupil_params": "pupil_params_placeholder",
                "cr_params": "cr_params_placeholder",
                "eye_params": "eye_params_placeholder"}

    monkeypatch.setattr(main, "load_ellipse_fit_params",
                        mock_load_ellipse_fit_params)

    obtained = main.preprocess_input_args(input_args)

    for key in expected.keys():
        if isinstance(obtained[key], np.ndarray):
            assert np.allclose(obtained[key], expected[key])
        else:
            assert obtained[key] == expected[key]


@pytest.mark.parametrize("pupil_params_rows, expected, expect_fail", [
    (5, pd.Series([1, 2, 3, 4, 5]), False),
    (4, None, True)
])
def test_load_sync_file_timings(monkeypatch, pupil_params_rows, expected, expect_fail):
    def mock_get_synchronized_frame_times(*args, **kwargs):
        return pd.Series([1, 2, 3, 4, 5])

    monkeypatch.setattr(main.su, "get_synchronized_frame_times",
                        mock_get_synchronized_frame_times)

    if expect_fail:
        with pytest.raises(RuntimeError, match="number of camera sync pulses"):
            main.load_sync_file_timings(Path("."), pupil_params_rows)

    else:
        obtained = main.load_sync_file_timings(Path("."), pupil_params_rows)
        assert expected.equals(obtained)
