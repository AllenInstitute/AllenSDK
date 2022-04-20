import json
import logging
import warnings
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from pynwb import NWBFile, TimeSeries

from allensdk.brain_observatory import sync_utilities
from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_files.eye_tracking_file import \
    EyeTrackingFile
from allensdk.core import DataObject
from allensdk.core import \
    NwbReadableInterface, DataFileReadableInterface
from allensdk.core import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.eye_tracking_processing import \
    process_eye_tracking_data, determine_outliers, determine_likely_blinks, \
    EyeTrackingError
from allensdk.brain_observatory.nwb.eye_tracking.ndx_ellipse_eye_tracking \
    import \
    EllipseSeries, EllipseEyeTracking
from allensdk.brain_observatory.sync_dataset import Dataset


class EyeTrackingTable(DataObject, DataFileReadableInterface,
                       NwbReadableInterface, NwbWritableInterface):
    """corneal, eye, and pupil ellipse fit data"""
    _logger = logging.getLogger(__name__)

    def __init__(self, eye_tracking: pd.DataFrame):
        super().__init__(name='eye_tracking', value=eye_tracking)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:

        # If there is actually no data in this data object,
        # do not bother writing anything to the NWBFile
        if self.value.empty:
            return nwbfile

        eye_tracking_df = self.value

        eye_tracking = EllipseSeries(
            name='eye_tracking',
            reference_frame='nose',
            data=eye_tracking_df[['eye_center_x', 'eye_center_y']].values,
            area=eye_tracking_df['eye_area'].values,
            area_raw=eye_tracking_df['eye_area_raw'].values,
            width=eye_tracking_df['eye_width'].values,
            height=eye_tracking_df['eye_height'].values,
            angle=eye_tracking_df['eye_phi'].values,
            timestamps=eye_tracking_df['timestamps'].values
        )

        pupil_tracking = EllipseSeries(
            name='pupil_tracking',
            reference_frame='nose',
            data=eye_tracking_df[['pupil_center_x', 'pupil_center_y']].values,
            area=eye_tracking_df['pupil_area'].values,
            area_raw=eye_tracking_df['pupil_area_raw'].values,
            width=eye_tracking_df['pupil_width'].values,
            height=eye_tracking_df['pupil_height'].values,
            angle=eye_tracking_df['pupil_phi'].values,
            timestamps=eye_tracking
        )

        corneal_reflection_tracking = EllipseSeries(
            name='corneal_reflection_tracking',
            reference_frame='nose',
            data=eye_tracking_df[['cr_center_x', 'cr_center_y']].values,
            area=eye_tracking_df['cr_area'].values,
            area_raw=eye_tracking_df['cr_area_raw'].values,
            width=eye_tracking_df['cr_width'].values,
            height=eye_tracking_df['cr_height'].values,
            angle=eye_tracking_df['cr_phi'].values,
            timestamps=eye_tracking
        )

        likely_blink = TimeSeries(timestamps=eye_tracking,
                                  data=eye_tracking_df['likely_blink'].values,
                                  name='likely_blink',
                                  description='blinks',
                                  unit='N/A')

        ellipse_eye_tracking = EllipseEyeTracking(
            eye_tracking=eye_tracking,
            pupil_tracking=pupil_tracking,
            corneal_reflection_tracking=corneal_reflection_tracking,
            likely_blink=likely_blink
        )

        nwbfile.add_acquisition(ellipse_eye_tracking)
        return nwbfile

    @classmethod
    def _get_empty_df(cls) -> pd.DataFrame:
        """
        Return an empty dataframe with the correct column and index
        names, but no data
        """
        empty_data = dict()
        for colname in ['timestamps', 'cr_area', 'eye_area',
                        'pupil_area', 'likely_blink', 'pupil_area_raw',
                        'cr_area_raw', 'eye_area_raw', 'cr_center_x',
                        'cr_center_y', 'cr_width', 'cr_height', 'cr_phi',
                        'eye_center_x', 'eye_center_y', 'eye_width',
                        'eye_height', 'eye_phi', 'pupil_center_x',
                        'pupil_center_y', 'pupil_width', 'pupil_height',
                        'pupil_phi']:
            empty_data[colname] = []

        eye_tracking_data = pd.DataFrame(empty_data,
                                         index=pd.Index([], name='frame'))
        return eye_tracking_data

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile,
                 z_threshold: float = 3.0,
                 dilation_frames: int = 2) -> Optional["EyeTrackingTable"]:
        """
        Parameters
        -----------
        nwbfile
        z_threshold
            See from_lims for description
        dilation_frames
            See from_lims for description
        """
        try:
            eye_tracking_acquisition = nwbfile.acquisition['EyeTracking']
        except KeyError as e:
            warnings.warn("This nwb file with identifier "
                          f"'{int(nwbfile.identifier)}' has no eye "
                          f"tracking data. (NWB error: {e})")
            eye_tracking_data = cls._get_empty_df()
            return EyeTrackingTable(eye_tracking=eye_tracking_data)

        eye_tracking = eye_tracking_acquisition.eye_tracking
        pupil_tracking = eye_tracking_acquisition.pupil_tracking
        corneal_reflection_tracking = \
            eye_tracking_acquisition.corneal_reflection_tracking

        eye_tracking_dict = {
            "timestamps": eye_tracking.timestamps[:],
            "cr_area": corneal_reflection_tracking.area_raw[:],
            "eye_area": eye_tracking.area_raw[:],
            "pupil_area": pupil_tracking.area_raw[:],
            "likely_blink": eye_tracking_acquisition.likely_blink.data[:],

            "pupil_area_raw": pupil_tracking.area_raw[:],
            "cr_area_raw": corneal_reflection_tracking.area_raw[:],
            "eye_area_raw": eye_tracking.area_raw[:],

            "cr_center_x": corneal_reflection_tracking.data[:, 0],
            "cr_center_y": corneal_reflection_tracking.data[:, 1],
            "cr_width": corneal_reflection_tracking.width[:],
            "cr_height": corneal_reflection_tracking.height[:],
            "cr_phi": corneal_reflection_tracking.angle[:],

            "eye_center_x": eye_tracking.data[:, 0],
            "eye_center_y": eye_tracking.data[:, 1],
            "eye_width": eye_tracking.width[:],
            "eye_height": eye_tracking.height[:],
            "eye_phi": eye_tracking.angle[:],

            "pupil_center_x": pupil_tracking.data[:, 0],
            "pupil_center_y": pupil_tracking.data[:, 1],
            "pupil_width": pupil_tracking.width[:],
            "pupil_height": pupil_tracking.height[:],
            "pupil_phi": pupil_tracking.angle[:],

        }

        eye_tracking_data = pd.DataFrame(eye_tracking_dict)
        eye_tracking_data.index = eye_tracking_data.index.rename('frame')

        # re-calculate likely blinks for new z_threshold and dilate_frames
        area_df = eye_tracking_data[['eye_area_raw', 'pupil_area_raw']]
        outliers = determine_outliers(area_df, z_threshold=z_threshold)
        likely_blinks = determine_likely_blinks(
            eye_tracking_data['eye_area_raw'],
            eye_tracking_data['pupil_area_raw'],
            outliers,
            dilation_frames=dilation_frames)

        eye_tracking_data["likely_blink"] = likely_blinks
        eye_tracking_data.loc[likely_blinks, "eye_area"] = np.nan
        eye_tracking_data.loc[likely_blinks, "pupil_area"] = np.nan
        eye_tracking_data.loc[likely_blinks, "cr_area"] = np.nan

        return EyeTrackingTable(eye_tracking=eye_tracking_data)

    @classmethod
    def from_data_file(cls, data_file: EyeTrackingFile,
                       sync_file: SyncFile,
                       drop_frames: Optional[List[int]] = None,
                       z_threshold: float = 3.0,
                       dilation_frames: int = 2,
                       ) -> "EyeTrackingTable":
        """
        Parameters
        ----------
        data_file
        sync_file
        drop_frames : List[int], optional
            List of frame indices to be dropped from the table.
            If provided, will drop the corresponding frame frame times read
            from the sync file to synchronize frame times and frames.
        z_threshold : float, optional
            See EyeTracking.from_lims
        dilation_frames : int, optional
             See EyeTracking.from_lims
        """
        cls._logger.info(f"Getting eye_tracking_data with "
                         f"'z_threshold={z_threshold}', "
                         f"'dilation_frames={dilation_frames}'")

        sync_path = Path(sync_file.filepath)

        frame_times = sync_utilities.get_synchronized_frame_times(
            session_sync_file=sync_path,
            sync_line_label_keys=Dataset.EYE_TRACKING_KEYS,
            drop_frames=drop_frames,
            trim_after_spike=False)

        try:
            eye_tracking_data = process_eye_tracking_data(
                                             data_file.data,
                                             frame_times,
                                             z_threshold,
                                             dilation_frames)
        except EyeTrackingError as err:
            msg = f"\nin sync_file: {sync_file.filepath}\n"
            msg += f"{str(err)}\n"
            msg += "returning empty eye_tracking DataFrame"
            warnings.warn(msg)
            eye_tracking_data = cls._get_empty_df()

        return EyeTrackingTable(eye_tracking=eye_tracking_data)


def get_lost_frames(file_path: str) -> List[int]:
    """
    Get lost frames from the video metadata json
    Must subtract one since the json starts indexing at 1

    Parameters
    ----------
    file_path: str
        Path to the metadata json

    Returns
    -------
        indices of lost frames
    """
    with open(file_path, 'r') as f:
        video_metadata = json.load(f)

    lost_frames = video_metadata['RecordingReport']['LostFrames']

    if lost_frames:
        return [int(f) - 1 for f in lost_frames]
    else:
        return []
