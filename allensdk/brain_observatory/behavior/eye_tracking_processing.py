from pathlib import Path

import numpy as np
import pandas as pd

from scipy import ndimage, stats


def load_eye_tracking_hdf(eye_tracking_file: Path) -> pd.DataFrame:
    """Load a DeepLabCut hdf5 file containing eye tracking data into a
    dataframe.

    Note: The eye tracking hdf5 file contains 3 separate dataframes. One for
    corneal reflection (cr), eye, and pupil ellipse fits. This function
    loads and returns this data as a single dataframe.

    Parameters
    ----------
    eye_tracking_file : Path
        Path to an hdf5 file produced by the DeepLabCut eye tracking pipeline.
        The hdf5 file will contain the following keys: "cr", "eye", "pupil".
        Each key has an associated dataframe with the following
        columns: "center_x", "center_y", "height", "width", "phi".

    Returns
    -------
    pd.DataFrame
        A dataframe containing combined corneal reflection (cr), eyelid (eye),
        and pupil data. Column names for each field will be renamed by
        prepending the field name. (e.g. center_x -> eye_center_x)
    """
    eye_tracking_fields = ["cr", "eye", "pupil"]

    eye_tracking_dfs = []
    for field_name in eye_tracking_fields:
        field_data = pd.read_hdf(eye_tracking_file, key=field_name)
        new_col_name_map = {col_name: f"{field_name}_{col_name}"
                            for col_name in field_data.columns}
        field_data.rename(new_col_name_map, axis=1, inplace=True)
        eye_tracking_dfs.append(field_data)

    eye_tracking_data = pd.concat(eye_tracking_dfs, axis=1)
    eye_tracking_data.index.name = 'frame'

    # Values in the hdf5 may be complex (likely an artifact of the ellipse
    # fitting process). Take only the real component.
    eye_tracking_data = eye_tracking_data.apply(lambda x: np.real(x.to_numpy()))

    return eye_tracking_data.astype(float)


def determine_outliers(data_df: pd.DataFrame,
                       z_threshold: float) -> pd.Series:
    """Given a dataframe and some z-score threshold return a pandas boolean
    Series where each entry indicates whether a given row contains at least
    one outlier (where outliers are calculated along columns).

    Parameters
    ----------
    data_df : pd.DataFrame
        A dataframe containing only columns where outlier detection is
        desired. (e.g. "cr_area", "eye_area", "pupil_area")
    z_threshold : float
        z-score values higher than the z_threshold will be considered outliers.

    Returns
    -------
    pd.Series
        A pandas boolean Series whose length == len(data_df.index).
        True denotes that a row in the data_df contains at least one outlier.
    """
    # Dataframe must have NANs filled to prevent warning when performing
    # comparisons against the z_threshold. NANs will be filtered out
    # in determine_likely_blinks().
    nan_filled_df = data_df.fillna(data_df.mean())

    outliers = (nan_filled_df.apply(stats.zscore).apply(np.abs) > z_threshold)
    return pd.Series(outliers.any(axis=1))


def compute_circular_area(df_row: pd.Series) -> float:
    """Calculate the area of the pupil as a circle using the max of the
    height/width as radius.

    Note: This calculation assumes that the pupil is a perfect circle
    and any eccentricity is a result of the angle at which the pupil is
    being viewed.

    Parameters
    ----------
    df_row : pd.Series
        A row from an eye tracking dataframe containing only "pupil_width"
        and "pupil_height".

    Returns
    -------
    float
        The circular area of the pupil in pixels^2.
    """
    max_dim = max(df_row.iloc[0], df_row.iloc[1])
    return np.pi * max_dim * max_dim


def compute_elliptical_area(df_row: pd.Series) -> float:
    """Calculate the area of corneal reflection (cr) or eye ellipse fits using
    the ellipse formula.

    Parameters
    ----------
    df_row : pd.Series
        A row from an eye tracking dataframe containing either:
        "cr_width", "cr_height"
        or
        "eye_width", "eye_height"

    Returns
    -------
    float
        The elliptical area of the eye or cr in pixels^2
    """
    return np.pi * df_row.iloc[0] * df_row.iloc[1]


def determine_likely_blinks(eye_areas: pd.Series,
                            pupil_areas: pd.Series,
                            outliers: pd.Series,
                            dilation_frames: int = 2) -> pd.Series:
    """Determine eye tracking frames which contain likely blinks or outliers

    Parameters
    ----------
    eye_areas : pd.Series
        A pandas series of eye areas.
    pupil_areas : pd.Series
        A pandas series of pupil areas.
    outliers : pd.Series
        A pandas series containing bool values of outlier rows.
    dilation_frames : int, optional
        Determines the number of additional adjacent frames to mark as
        'likely_blink', by default 2.

    Returns
    -------
    pd.Series
        A pandas series of bool values that has the same length as the number
        of eye tracking dataframe rows (frames).
    """
    blinks = pd.isnull(eye_areas) | pd.isnull(pupil_areas) | outliers
    if dilation_frames > 0:
        likely_blinks = ndimage.binary_dilation(blinks,
                                                iterations=dilation_frames)
    else:
        likely_blinks = blinks
    return pd.Series(likely_blinks)


def process_eye_tracking_data(eye_data: pd.DataFrame,
                              frame_times: pd.Series,
                              z_threshold: float = 3.0,
                              dilation_frames: int = 2) -> pd.DataFrame:
    """Processes and refines raw eye tracking data by adding additional
    computed feature columns.

    Parameters
    ----------
    eye_data : pd.DataFrame
        A 'raw' eye tracking dataframe produced by load_eye_tracking_hdf()
    frame_times : pd.Series
        A series of frame times acquired from a behavior + ophy session
        'sync file'.
    z_threshold : float
        z-score values higher than the z_threshold will be considered outliers,
        by default 3.0.
    dilation_frames : int, optional
        Determines the number of additional adjacent frames to mark as
        'likely_blink', by default 2.

    Returns
    -------
    pd.DataFrame
        A refined eye tracking dataframe that contains additional information
        about frame times, eye areas, pupil areas, and frames with likely
        blinks/outliers.

    Raises
    ------
    RuntimeError
        If the number of sync file frame times does not match the number of
        eye tracking frames.
    """

    if len(frame_times) != len(eye_data.index):
        raise RuntimeError(f"Error! The number of sync file frame times "
                           f"({len(frame_times)} does not match the "
                           f"number of eye tracking frames "
                           f"({len(eye_data.index)})!")

    cr_areas = (eye_data[["cr_width", "cr_height"]]
                .apply(compute_elliptical_area, axis=1))
    eye_areas = (eye_data[["eye_width", "eye_height"]]
                 .apply(compute_elliptical_area, axis=1))
    pupil_areas = (eye_data[["pupil_width", "pupil_height"]]
                   .apply(compute_circular_area, axis=1))

    area_df = pd.concat([cr_areas, eye_areas, pupil_areas], axis=1)
    outliers = determine_outliers(area_df, z_threshold=z_threshold)

    likely_blinks = determine_likely_blinks(eye_areas,
                                            pupil_areas,
                                            outliers,
                                            dilation_frames=dilation_frames)

    eye_data.insert(0, "time", frame_times)
    eye_data.insert(1, "cr_area", cr_areas)
    eye_data.insert(2, "eye_area", eye_areas)
    eye_data.insert(3, "pupil_area", pupil_areas)
    eye_data.insert(4, "likely_blink", likely_blinks)

    return eye_data
