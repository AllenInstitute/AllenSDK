import logging
from typing import Optional

import numpy as np
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    SyncFileReadableInterface, NwbReadableInterface


class OphysTimestamps(DataObject, SyncFileReadableInterface,
                      NwbReadableInterface):
    _logger = logging.getLogger(__name__)

    def __init__(self, timestamps: np.ndarray,
                 validate=True,
                 number_of_frames: Optional[int] = None):
        """
        :param timestamps
            ophys timestamps
        :param validate
            Whether to validate timestamps
        :param number_of_frames
            number of frames in the movie. Used for validation
        """
        if validate:
            if number_of_frames is None:
                raise ValueError(
                    'Need number of frames to validate timestamps')
            timestamps = self._validate(ophys_timestamps=timestamps,
                                        number_of_frames=number_of_frames)
        super().__init__(name='ophys_timestamps', value=timestamps)

    @classmethod
    def from_sync_file(cls, sync_file: SyncFile,
                        number_of_frames: int) -> "OphysTimestamps":
        ophys_timestamps = sync_file.data['ophys_frames']
        return cls(timestamps=ophys_timestamps,
                   number_of_frames=number_of_frames)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "OphysTimestamps":
        ts = nwbfile.processing[
                   'ophys'].get_data_interface('dff').roi_response_series[
                   'traces'].timestamps[:]
        return cls(timestamps=ts, validate=False)

    def _validate(self, number_of_frames: int,
                 ophys_timestamps: np.ndarray) -> np.ndarray:
        """Validates that number of ophys timestamps do not exceed number of
        dff traces. If so, truncates number of ophys timestamps to the same
        length as dff traces

        :param number_of_frames
            number of frames in the movie
        :param ophys_timestamps
            ophys timestamps
        """
        # Scientifica data has extra frames in the sync file relative
        # to the number of frames in the video. These sentinel frames
        # should be removed.
        # NOTE: This fix does not apply to mesoscope data.
        # See http://confluence.corp.alleninstitute.org/x/9DVnAg
        num_of_timestamps = len(ophys_timestamps)
        if number_of_frames < num_of_timestamps:
            self._logger.info(
                "Truncating acquisition frames ('ophys_frames') "
                f"(len={num_of_timestamps}) to the number of frames "
                f"in the df/f trace ({number_of_frames}).")
            ophys_timestamps = ophys_timestamps[:number_of_frames]
        elif number_of_frames > num_of_timestamps:
            raise RuntimeError(
                f"dff_frames (len={number_of_frames}) is longer "
                f"than timestamps (len={num_of_timestamps}).")
        return ophys_timestamps


class OphysTimestampsMultiplane(OphysTimestamps):
    def __init__(self, timestamps: np.ndarray, number_of_frames: int):
        super().__init__(timestamps=timestamps,
                         number_of_frames=number_of_frames)

    @classmethod
    def from_sync_file(cls, sync_file: SyncFile,
                       number_of_frames: int,
                       group_count: int,
                       plane_group: int) -> "OphysTimestampsMultiplane":
        ophys_timestamps = sync_file.data['ophys_frames']
        cls._logger.info(
            "Mesoscope data detected. Splitting timestamps "
            f"(len={len(ophys_timestamps)} over {group_count} "
            "plane group(s).")

        # Resample if collecting multiple concurrent planes
        # because the frames are interleaved
        ophys_timestamps = ophys_timestamps[plane_group::group_count]

        return cls(timestamps=ophys_timestamps,
                   number_of_frames=number_of_frames)

    def _validate(self, number_of_frames: int,
                 ophys_timestamps: np.ndarray) -> np.ndarray:
        """
        Raises error if length of timestamps and number of frames are not equal
        :param number_of_frames
            See super()._validate
        :param ophys_timestamps
            See super()._validate
        """
        num_of_timestamps = len(ophys_timestamps)
        if number_of_frames != num_of_timestamps:
            raise RuntimeError(
                f"dff_frames (len={number_of_frames}) is not equal to "
                f"number of split timestamps (len={num_of_timestamps}).")
        return ophys_timestamps
