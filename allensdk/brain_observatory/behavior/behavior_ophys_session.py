import numpy as np
import pandas as pd
from typing import Any

from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.session_api_utils import ParamsMixin
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi, BehaviorOphysLimsApi)
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.image_api import Image, ImageApi


class BehaviorOphysSession(BehaviorSession, ParamsMixin):
    """Represents data from a single Visual Behavior Ophys imaging session.
    Can be initialized with an api that fetches data, or by using class methods
    `from_lims` and `from_nwb_path`.
    """

    def __init__(self, api=None,
                 eye_tracking_z_threshold: float = 3.0,
                 eye_tracking_dilation_frames: int = 2,
                 events_filter_scale: float = 2.0,
                 events_filter_n_time_steps: int = 20):
        """
        Parameters
        ----------
        api : object, optional
            The backend api used by the session object to get behavior ophys
            data, by default None.
        eye_tracking_z_threshold : float, optional
            The z-threshold when determining which frames likely contain
            outliers for eye or pupil areas. Influences which frames
            are considered 'likely blinks'. By default 3.0
        eye_tracking_dilation_frames : int, optional
            Determines the number of adjacent frames that will be marked
            as 'likely_blink' when performing blink detection for
            `eye_tracking` data, by default 2
        events_filter_scale : float, optional
            Stdev of halfnorm distribution used to convolve ophys events with
            a 1d causal half-gaussian filter to smooth it for visualization,
            by default 2.0
        events_filter_n_time_steps : int, optional
            Number of time steps to use for convolution of ophys events
        """

        BehaviorSession.__init__(self, api=api)
        ParamsMixin.__init__(self, ignore={'api'})

        # eye_tracking processing params
        self._eye_tracking_z_threshold = eye_tracking_z_threshold
        self._eye_tracking_dilation_frames = eye_tracking_dilation_frames

        # events processing params
        self._events_filter_scale = events_filter_scale
        self._events_filter_n_time_steps = events_filter_n_time_steps

        # LazyProperty constructor provided by LazyPropertyMixin
        LazyProperty = self.LazyProperty

        # Initialize attributes to be lazily evaluated
        self._ophys_session_id = LazyProperty(
            self.api.get_ophys_session_id)
        self._ophys_experiment_id = LazyProperty(
            self.api.get_ophys_experiment_id)
        self._max_projection = LazyProperty(self.api.get_max_projection,
                                            wrappers=[ImageApi.deserialize])
        self._average_projection = LazyProperty(
            self.api.get_average_projection, wrappers=[ImageApi.deserialize])
        self._ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps,
                                              settable=True)
        self._dff_traces = LazyProperty(self.api.get_dff_traces, settable=True)
        self._events = LazyProperty(self.api.get_events, settable=True)
        self._cell_specimen_table = LazyProperty(
            self.api.get_cell_specimen_table, settable=True)
        self._corrected_fluorescence_traces = LazyProperty(
            self.api.get_corrected_fluorescence_traces, settable=True)
        self._motion_correction = LazyProperty(self.api.get_motion_correction,
                                               settable=True)
        self._segmentation_mask_image = LazyProperty(
            self.get_segmentation_mask_image)
        self._eye_tracking = LazyProperty(
            self.api.get_eye_tracking, settable=True,
            z_threshold=self._eye_tracking_z_threshold,
            dilation_frames=self._eye_tracking_dilation_frames)
        self._eye_tracking_rig_geometry = LazyProperty(
             self.api.get_eye_tracking_rig_geometry)

    # ==================== class and utility methods ======================

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  eye_tracking_z_threshold: float = 3.0,
                  eye_tracking_dilation_frames: int = 2
                  ) -> "BehaviorOphysSession":
        return cls(api=BehaviorOphysLimsApi(ophys_experiment_id),
                   eye_tracking_z_threshold=eye_tracking_z_threshold,
                   eye_tracking_dilation_frames=eye_tracking_dilation_frames)

    @classmethod
    def from_nwb_path(
            cls, nwb_path: str, **api_kwargs: Any) -> "BehaviorOphysSession":
        api_kwargs["filter_invalid_rois"] = api_kwargs.get(
            "filter_invalid_rois", True)
        return cls(api=BehaviorOphysNwbApi.from_path(
            path=nwb_path, **api_kwargs))

    # ========================= 'get' methods ==========================

    def get_segmentation_mask_image(self):
        """ Returns an image with value 1 if the pixel was included
        in an ROI, and 0 otherwise

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to segmentation_mask image data and metadata
        """
        mask_data = np.sum(self.roi_masks['roi_mask'].values).astype(int)

        max_projection_image = self.max_projection

        mask_image = Image(
            data=mask_data,
            spacing=max_projection_image.spacing,
            unit=max_projection_image.unit
        )
        return mask_image

    @legacy('Consider using "dff_traces" instead.')
    def get_dff_traces(self, cell_specimen_ids=None):

        if cell_specimen_ids is None:
            cell_specimen_ids = self.get_cell_specimen_ids()

        csid_table = \
            self.cell_specimen_table.reset_index()[['cell_specimen_id']]
        csid_subtable = csid_table[csid_table['cell_specimen_id'].isin(
            cell_specimen_ids)].set_index('cell_specimen_id')
        dff_table = csid_subtable.join(self.dff_traces, how='left')
        dff_traces = np.vstack(dff_table['dff'].values)
        timestamps = self.ophys_timestamps

        assert (len(cell_specimen_ids), len(timestamps)) == dff_traces.shape
        return timestamps, dff_traces

    @legacy()
    def get_cell_specimen_indices(self, cell_specimen_ids):
        return [self.cell_specimen_table.index.get_loc(csid)
                for csid in cell_specimen_ids]

    @legacy("Consider using cell_specimen_table['cell_specimen_id'] instead.")
    def get_cell_specimen_ids(self):
        cell_specimen_ids = self.cell_specimen_table.index.values

        if np.isnan(cell_specimen_ids.astype(float)).sum() == \
                len(self.cell_specimen_table):
            raise ValueError("cell_specimen_id values not assigned "
                             f"for {self.ophys_experiment_id}")
        return cell_specimen_ids

    # ====================== properties and setters ========================

    @property
    def ophys_experiment_id(self) -> int:
        """Unique identifier for this experimental session.
        :rtype: int
        """
        return self._ophys_experiment_id

    @property
    def max_projection(self) -> Image:
        """2D max projection image.
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        return self._max_projection

    @property
    def average_projection(self) -> pd.DataFrame:
        """2D image of the microscope field of view, averaged across the
        experiment
        :rtype: pandas.DataFrame
        """
        return self._average_projection

    @property
    def ophys_timestamps(self) -> np.ndarray:
        """Timestamps associated with frames captured by the microscope
        :rtype: numpy.ndarray
        """
        return self._ophys_timestamps

    @ophys_timestamps.setter
    def ophys_timestamps(self, value):
        self._ophys_timestamps = value

    @property
    def dff_traces(self) -> pd.DataFrame:
        """Traces of dff organized into a dataframe; index is the cell roi ids.
        :rtype: pandas.DataFrame
        """
        return self._dff_traces

    @dff_traces.setter
    def dff_traces(self, value):
        self._dff_traces = value

    @property
    def events(self) -> pd.DataFrame:
        """Get event detection data

        Returns
        -------
        pd.DataFrame
            index:
                cell_specimen_id: int
            cell_roi_id: int
            events: np.array
            filtered_events: np.array
                Events, convolved with filter to smooth it for visualization
            lambdas: float64
            noise_stds: float64
        """
        params = {'events_filter_scale', 'events_filter_n_time_steps'}

        if self.needs_data_refresh(params):
            self._events = self.LazyProperty(
                self.api.get_events,
                filter_scale=self._events_filter_scale,
                filter_n_time_steps=self._events_filter_n_time_steps)
            self.clear_updated_params(params)

        return self._events

    @events.setter
    def events(self, value):
        self_events = value

    @property
    def cell_specimen_table(self) -> pd.DataFrame:
        """Cell roi information organized into a dataframe; index is the cell
        roi ids.
        :rtype: pandas.DataFrame
        """
        return self._cell_specimen_table

    @cell_specimen_table.setter
    def cell_specimen_table(self, value):
        self._cell_specimen_table = value

    @property
    def corrected_fluorescence_traces(self) -> pd.DataFrame:
        """The motion-corrected fluorescence traces organized into a dataframe;
        index is the cell roi ids.
        :rtype: pandas.DataFrame
        """
        return self._corrected_fluorescence_traces

    @corrected_fluorescence_traces.setter
    def corrected_fluorescence_traces(self, value):
        self._corrected_fluorescence_traces = value

    @property
    def motion_correction(self) -> pd.DataFrame:
        """A dataframe containing trace data used during motion correction
        computation
        :rtype: pandas.DataFrame
        """
        return self._motion_correction

    @motion_correction.setter
    def motion_correction(self, value):
        self._motion_correction = value

    @property
    def segmentation_mask_image(self) -> Image:
        """An image with pixel value 1 if that pixel was included in an ROI,
        and 0 otherwise
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        if self._segmentation_mask_image is None:
            self._segmentation_mask_image = self.get_segmentation_mask_image()
        return self._segmentation_mask_image

    @segmentation_mask_image.setter
    def segmentation_mask_image(self, value):
        self._segmentation_mask_image = value

    @property
    def eye_tracking(self) -> pd.DataFrame:
        """A dataframe containing ellipse fit parameters for the eye, pupil
        and corneal reflection (cr). Fits are derived from tracking points
        from a DeepLabCut model applied to video frames of a subject's
        right eye. Raw tracking points and raw video frames are not exposed
        by the SDK.

        Notes:
        - All columns starting with 'pupil_' represent ellipse fit parameters
          relating to the pupil.
        - All columns starting with 'eye_' represent ellipse fit parameters
          relating to the eyelid.
        - All columns starting with 'cr_' represent ellipse fit parameters
          relating to the corneal reflection, which is caused by an infrared
          LED positioned near the eye tracking camera.
        - All positions are in units of pixels.
        - All areas are in units of pixels^2
        - All values are in the coordinate space of the eye tracking camera,
          NOT the coordinate space of the stimulus display (i.e. this is not
          gaze location), with (0, 0) being the upper-left corner of the
          eye-tracking image.
        - The 'likely_blink' column is True for any row (frame) where the pupil
          fit failed OR eye fit failed OR an outlier fit was identified on the pupil or eye fit.
        - The pupil_area, cr_area, eye_area columns are set to NaN wherever 'likely_blink' == True.
        - The pupil_area_raw, cr_area_raw, eye_area_raw columns contains all pupil fit values (including where 'likely_blink' == True).
        - All ellipse fits are derived from tracking points that were output by
          a DeepLabCut model that was trained on hand-annotated data frome a
          subset of imaging sessions on optical physiology rigs.
        - Raw DeepLabCut tracking points are not publicly available.

        :rtype: pandas.DataFrame
        """
        params = {'eye_tracking_dilation_frames', 'eye_tracking_z_threshold'}

        if self.needs_data_refresh(params):
            self._eye_tracking = self.LazyProperty(
                self.api.get_eye_tracking,
                z_threshold=self._eye_tracking_z_threshold,
                dilation_frames=self._eye_tracking_dilation_frames)
            self.clear_updated_params(params)

        return self._eye_tracking

    @eye_tracking.setter
    def eye_tracking(self, value):
        self._eye_tracking = value

    @property
    def eye_tracking_rig_geometry(self) -> dict:
        """Get the eye tracking rig geometry
        associated with an ophys experiment"""
        return self.api.get_eye_tracking_rig_geometry()

    @property
    def roi_masks(self) -> pd.DataFrame:
        return self.cell_specimen_table[['cell_roi_id', 'roi_mask']]


if __name__ == "__main__":

    ophys_experiment_id = 789359614
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)
    print(session.trials['reward_time'])
