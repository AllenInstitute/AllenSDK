import abc
from typing import Optional, Union

import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.metadata.behavior_ophys_metadata \
    import BehaviorOphysMetadata
from allensdk.brain_observatory.behavior.session_apis.abcs.\
    session_base.behavior_base import BehaviorBase
from allensdk.brain_observatory.behavior.image_api import Image


class BehaviorOphysBase(BehaviorBase):
    """Abstract base class implementing required methods for interacting with
    behavior + ophys session data.

    Child classes should be instantiated with a fetch API that implements these
    methods.
    """

    @abc.abstractmethod
    def get_ophys_experiment_id(self) -> Optional[int]:
        """Returns the ophys_experiment_id for the instantiated BehaviorOphys
        Session (or BehaviorOphys data fetcher) if applicable."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ophys_session_id(self) -> Optional[int]:
        """Returns the behavior + ophys_session_id associated with this
        experiment, if applicable.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_average_projection(self) -> Image:
        """Get an image whose values are the average obtained values at
        each pixel of the ophys movie over time.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            Array-like interface to avg projection image data and metadata.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_max_projection(self) -> Image:
        """Get an image whose values are the maximum obtained values at
        each pixel of the ophys movie over time.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            Array-like interface to max projection image data and metadata.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_cell_specimen_table(self) -> pd.DataFrame:
        """Get a cell specimen dataframe containing ROI information about
        cells identified in an ophys experiment.

        Returns
        -------
        pd.DataFrame
            Cell ROI information organized into a dataframe.
            Index is the cell ROI IDs.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_corrected_fluorescence_traces(self) -> pd.DataFrame:
        """Get motion-corrected fluorescence traces.

        Returns
        -------
        pd.DataFrame
            Motion-corrected fluorescence traces organized into a dataframe.
            Index is the cell ROI IDs.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dff_traces(self) -> pd.DataFrame:
        """Get a table of delta fluorescence over fluorescence traces.

        Returns
        -------
        pd.DataFrame
            The traces of dff (normalized fluorescence) organized into a
            dataframe. Index is the cell ROI IDs.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_metadata(self) -> Union[BehaviorOphysMetadata, dict]:
        """Get behavior+ophys session metadata.

        Returns
        -------
        dict if NWB
        BehaviorOphysMetadata otherwise
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_motion_correction(self) -> pd.DataFrame:
        """Get motion correction trace data.

        Returns
        -------
        pd.DataFrame
             A dataframe containing trace data used during motion
             correction computation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ophys_timestamps(self) -> np.ndarray:
        """Get optical physiology frame timestamps.

        Returns
        -------
        np.ndarray
            Timestamps associated with frames captured by the microscope.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stimulus_timestamps(self) -> np.ndarray:
        """Get stimulus timestamps.

        Returns
        -------
        np.ndarray
            Timestamps associated with stimulus presentations on the monitor
            after accounting for monitor delay.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stimulus_presentations(self) -> pd.DataFrame:
        """Get stimulus presentation data.

        NOTE: Uses monitor delay corrected stimulus timestamps.

        Returns
        -------
        pd.DataFrame
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_eye_tracking(self) -> Optional[pd.DataFrame]:
        """Get eye tracking data from behavior + ophys session.

        Returns
        -------
        pd.DataFrame
            A refined eye tracking dataframe that contains information
            about eye tracking ellipse fits, frame times, eye areas,
            pupil areas, and frames with likely blinks/outliers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_events(self) -> pd.DataFrame:
        """Get event detection data

        Returns
        -------
        pd.DataFrame
            index:
                cell_specimen_id: int
            cell_roi_id: int
            events: np.array
            filtered_events: np.array
            lambdas: float64
            noise_stds: float64
        """
        raise NotImplementedError()

    def get_eye_tracking_rig_geometry(self) -> dict:
        """Get eye tracking rig metadata from behavior + ophys session.

        Returns
        -------
        dict
            Includes geometry of monitor, camera, LED
        """
        raise NotImplementedError()
