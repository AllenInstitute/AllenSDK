from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.eye_tracking_processing import (
    load_eye_tracking_hdf, determine_outliers, compute_circular_area,
    compute_elliptical_area, determine_likely_blinks, process_eye_tracking_data)


def create_preload_eye_tracking_df(data: np.ndarray) -> pd.DataFrame:
    columns = ["center_x", "center_y", "width", "height", "phi"]
    return pd.DataFrame(data, columns=columns)


def create_loaded_eye_tracking_df(data: np.ndarray) -> pd.DataFrame:
    columns = ["cr_center_x", "cr_center_y", "cr_width", "cr_height", "cr_phi",
               "eye_center_x", "eye_center_y", "eye_width", "eye_height",
               "eye_phi", "pupil_center_x", "pupil_center_y", "pupil_width",
               "pupil_height", "pupil_phi"]
    df = pd.DataFrame(data, columns=columns)
    df.index.name = 'frame'
    return df


def create_area_df(data: np.ndarray) -> pd.DataFrame:
    columns = ["cr_area", "eye_area", "pupil_area"]
    return pd.DataFrame(data, columns=columns)


def create_refined_eye_tracking_df(data: np.ndarray) -> pd.DataFrame:
    columns = ["time", "cr_area", "eye_area", "pupil_area", "likely_blink",
               "cr_center_x", "cr_center_y", "cr_width", "cr_height", "cr_phi",
               "eye_center_x", "eye_center_y", "eye_width", "eye_height",
               "eye_phi", "pupil_center_x", "pupil_center_y", "pupil_width",
               "pupil_height", "pupil_phi"]
    df = pd.DataFrame(data, columns=columns)
    df.index.name = 'frame'
    # Initializing a df coerces all data to one dtype
    # restoring the bool dtype for the 'likely_blink' column.
    df['likely_blink'] = df['likely_blink'].apply(bool)
    return df


@pytest.fixture
def hdf_fixture(request, tmp_path) -> Path:
    """Creates a mock eye tracking h5 file to test loading functionality"""
    tmp_hdf_path = tmp_path / "mock_eye_tracking_ellipse_fits.h5"

    test_data = request.param
    cr = create_preload_eye_tracking_df(test_data["cr"])
    eye = create_preload_eye_tracking_df(test_data["eye"])
    pupil = create_preload_eye_tracking_df(test_data["pupil"])

    cr.to_hdf(tmp_hdf_path, key="cr", mode="w")
    eye.to_hdf(tmp_hdf_path, key="eye", mode="a")
    pupil.to_hdf(tmp_hdf_path, key="pupil", mode="a")

    return tmp_hdf_path


@pytest.mark.parametrize("hdf_fixture, expected", [
    ({"cr": np.array([[1., 2., 3., 4., 5.]]),
      "eye": np.array([[6., 7., 8., 9., 10.]]),
      "pupil": np.array([[11., 12., 13., 14., 15.]])},

     create_loaded_eye_tracking_df(
        np.array([[1., 2., 3., 4., 5., 6., 7., 8.,
                   9., 10., 11., 12., 13., 14., 15.]]))
     ),

    ({"cr": np.array([[5 + 2j, 4 + 1j, 3 + 1j, 2 + 8j, 1 + 1j]]),
      "eye": np.array([[6, 7, 8, 9, 10]]),
      "pupil": np.array([[15 + 1j, 14 + 3j, 13 + 2j, 12 + 1j, 11 + 1j]])},

     create_loaded_eye_tracking_df(
        np.array([[5., 4., 3., 2., 1., 6., 7., 8.,
                   9., 10., 15., 14., 13., 12., 11.]]))
     ),

], indirect=["hdf_fixture"])
def test_load_eye_tracking_hdf(hdf_fixture: Path, expected: pd.DataFrame):
    obtained = load_eye_tracking_hdf(hdf_fixture)
    assert expected.equals(obtained)


@pytest.mark.parametrize("data_df, z_threshold, expected", [
    (create_area_df(
        np.array([[1, 1, 2],
                  [2, 2, 1],
                  [1, 7, 3],
                  [1, 1, 1],
                  [1, 3, 2],
                  [1, 1, 1],
                  [1, 2, 1],
                  [2, 1, 1000]])),
     2.5,
     pd.Series([False, False, False, False, False, False, False, True])),

    (create_area_df(
        np.array([[1, 1, 2],
                  [2, 2, 1],
                  [1, 7, 3],
                  [1, 1, 1],
                  [1, 3, 2],
                  [1, 1, 1],
                  [1, 2, 1],
                  [2, 1, 1000]])),
     2.0,
     pd.Series([False, False, True, False, False, False, False, True])),

])
def test_determine_outliers(data_df, z_threshold, expected):
    obtained = determine_outliers(data_df, z_threshold)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize("df_row, expected", [
    (pd.Series([3, 2], index=["width", "height"]), 9 * np.pi),
    (pd.Series([2, 3], index=["width", "height"]), 9 * np.pi),
])
def test_compute_circular_area(df_row: pd.Series, expected: float):
    obtained_area = compute_circular_area(df_row)
    assert obtained_area == expected


@pytest.mark.parametrize("df_row, expected", [
    (pd.Series([3, 2], index=["width", "height"]), 6 * np.pi),
    (pd.Series([2, 3], index=["width", "height"]), 6 * np.pi),
])
def test_compute_elliptical_area(df_row: pd.Series, expected: float):
    obtained_area = compute_elliptical_area(df_row)
    assert obtained_area == expected


@pytest.mark.parametrize("eye_areas, pupil_areas, outliers, dilation_frames, expected", [
    (pd.Series([4, 8, 3, 20, np.nan, 10, 21, 19, 42]),
     pd.Series([np.nan, 10, 2, 30, 99, 80, 93, 18, 777]),
     pd.Series([False, False, False, False, False, False, False, False, True]),
     2,
     pd.Series([True, True, True, True, True, True, True, True, True])),

    (pd.Series([4, 8, 3, 20, np.nan, 10, 21, 19, 42]),
     pd.Series([np.nan, 10, 2, 30, 99, 80, 93, 18, 777]),
     pd.Series([False, False, False, False, False, False, False, False, True]),
     1,
     pd.Series([True, True, False, True, True, True, False, True, True])),


    (pd.Series([4, 8, 3, 20, np.nan, 10, 21, 19, 42]),
     pd.Series([np.nan, 10, 2, 30, 99, 80, 93, 18, 777]),
     pd.Series([False, False, False, False, False, False, False, False, True]),
     0,
     pd.Series([True, False, False, False, True, False, False, False, True])),
])
def test_determine_likely_blinks(eye_areas, pupil_areas, outliers,
                                 dilation_frames, expected):
    obtained = determine_likely_blinks(eye_areas, pupil_areas, outliers,
                                       dilation_frames)
    assert expected.equals(obtained)


@pytest.mark.parametrize("eye_tracking_df, frame_times", [
    (create_loaded_eye_tracking_df(
        np.array([[1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1],
                  [2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2]])),
     pd.Series([0.1, 0.2, 0.3])),
])
def test_process_eye_tracking_data_raises_on_sync_error(eye_tracking_df,
                                                        frame_times):
    with pytest.raises(RuntimeError, match='Error! The number of sync file'):
        _ = process_eye_tracking_data(eye_tracking_df, frame_times)


@pytest.mark.parametrize("eye_tracking_df, frame_times, expected", [
    (create_loaded_eye_tracking_df(
        np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.],
                  [2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]])),
     pd.Series([0.1, 0.2]),
     create_refined_eye_tracking_df(
         np.array([[0.1, 12 * np.pi, 72 * np.pi, 196 * np.pi, False,
                    1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.],
                   [0.2, 20 * np.pi, 90 * np.pi, 225 * np.pi, False,
                    2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]]))
     ),
])
def test_process_eye_tracking_data(eye_tracking_df, frame_times, expected):
    obtained = process_eye_tracking_data(eye_tracking_df, frame_times)
    assert expected.equals(obtained)
