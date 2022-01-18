from typing import Optional

import numpy as np
import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_files.eye_tracking_file import \
    EyeTrackingFile
from allensdk.brain_observatory.behavior.data_files\
    .rigid_motion_transform_file import \
    RigidMotionTransformFile
from allensdk.brain_observatory.behavior.data_objects import \
    BehaviorSessionId, StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.cell_specimens \
    .cell_specimens import \
    CellSpecimens, EventsParams
from allensdk.brain_observatory.behavior.data_objects.eye_tracking\
    .eye_tracking_table import \
    EyeTrackingTable
from allensdk.brain_observatory.behavior.data_objects.eye_tracking\
    .rig_geometry import \
    RigGeometry as EyeTrackingRigGeometry
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisitionOphys, DateOfAcquisition
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_ophys_metadata import \
    BehaviorOphysMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.multi_plane_metadata.imaging_plane_group \
    import \
    ImagingPlaneGroup
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .ophys_experiment_metadata.multi_plane_metadata.multi_plane_metadata \
    import \
    MultiplaneMetadata
from allensdk.brain_observatory.behavior.data_objects.motion_correction \
    import \
    MotionCorrection
from allensdk.brain_observatory.behavior.data_objects.projections import \
    Projections
from allensdk.brain_observatory.behavior.data_objects.stimuli.util import \
    calculate_monitor_delay
from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .ophys_timestamps import \
    OphysTimestamps, OphysTimestampsMultiplane
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.image_api import Image
from allensdk.internal.api import db_connection_creator


class BehaviorOphysExperiment(BehaviorSession):
    """Represents data from a single Visual Behavior Ophys imaging session.
    Initialize by using class methods `from_lims` or `from_nwb_path`.
    """

    def __init__(self,
                 behavior_session: BehaviorSession,
                 projections: Projections,
                 ophys_timestamps: OphysTimestamps,
                 cell_specimens: CellSpecimens,
                 metadata: BehaviorOphysMetadata,
                 motion_correction: MotionCorrection,
                 eye_tracking_table: Optional[EyeTrackingTable],
                 eye_tracking_rig_geometry: Optional[EyeTrackingRigGeometry],
                 date_of_acquisition: DateOfAcquisition):
        super().__init__(
            behavior_session_id=behavior_session._behavior_session_id,
            licks=behavior_session._licks,
            metadata=behavior_session._metadata,
            raw_running_speed=behavior_session._raw_running_speed,
            rewards=behavior_session._rewards,
            running_speed=behavior_session._running_speed,
            running_acquisition=behavior_session._running_acquisition,
            stimuli=behavior_session._stimuli,
            stimulus_timestamps=behavior_session._stimulus_timestamps,
            task_parameters=behavior_session._task_parameters,
            trials=behavior_session._trials,
            date_of_acquisition=date_of_acquisition
        )

        self._metadata = metadata
        self._projections = projections
        self._ophys_timestamps = ophys_timestamps
        self._cell_specimens = cell_specimens
        self._motion_correction = motion_correction
        self._eye_tracking = eye_tracking_table
        self._eye_tracking_rig_geometry = eye_tracking_rig_geometry

    def to_nwb(self) -> NWBFile:
        nwbfile = super().to_nwb(add_metadata=False)

        self._metadata.to_nwb(nwbfile=nwbfile)
        self._projections.to_nwb(nwbfile=nwbfile)
        self._cell_specimens.to_nwb(nwbfile=nwbfile,
                                    ophys_timestamps=self._ophys_timestamps)
        self._motion_correction.to_nwb(nwbfile=nwbfile)
        self._eye_tracking.to_nwb(nwbfile=nwbfile)
        if self._eye_tracking_rig_geometry is not None:
            self._eye_tracking_rig_geometry.to_nwb(nwbfile=nwbfile)

        return nwbfile
    # ==================== class and utility methods ======================

    @classmethod
    def from_lims(cls,
                  ophys_experiment_id: int,
                  eye_tracking_z_threshold: float = 3.0,
                  eye_tracking_dilation_frames: int = 2,
                  events_filter_scale_seconds: float = 2.0/31.0,
                  events_filter_n_time_steps: int = 20,
                  exclude_invalid_rois=True,
                  skip_eye_tracking=False) -> \
            "BehaviorOphysExperiment":
        """
        Parameters
        ----------
        ophys_experiment_id
        eye_tracking_z_threshold
            See `BehaviorOphysExperiment.from_nwb`
        eye_tracking_dilation_frames
            See `BehaviorOphysExperiment.from_nwb`
        events_filter_scale_seconds
            See `BehaviorOphysExperiment.from_nwb`
        events_filter_n_time_steps
            See `BehaviorOphysExperiment.from_nwb`
        exclude_invalid_rois
            Whether to exclude invalid rois
        skip_eye_tracking
            Used to skip returning eye tracking data
        """
        def _is_multi_plane_session():
            imaging_plane_group_meta = ImagingPlaneGroup.from_lims(
                ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
            return cls._is_multi_plane_session(
                imaging_plane_group_meta=imaging_plane_group_meta)

        def _get_motion_correction():
            rigid_motion_transform_file = RigidMotionTransformFile.from_lims(
                ophys_experiment_id=ophys_experiment_id, db=lims_db
            )
            return MotionCorrection.from_data_file(
                rigid_motion_transform_file=rigid_motion_transform_file)

        def _get_eye_tracking_table(sync_file: SyncFile):
            eye_tracking_file = EyeTrackingFile.from_lims(
                db=lims_db, ophys_experiment_id=ophys_experiment_id)
            eye_tracking_table = EyeTrackingTable.from_data_file(
                data_file=eye_tracking_file,
                sync_file=sync_file,
                z_threshold=eye_tracking_z_threshold,
                dilation_frames=eye_tracking_dilation_frames
            )
            return eye_tracking_table

        lims_db = db_connection_creator(
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP
        )

        is_multiplane_session = _is_multi_plane_session()

        meta = BehaviorOphysMetadata.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db,
            is_multiplane=is_multiplane_session
        )

        sync_file = SyncFile.from_lims(db=lims_db,
                                       ophys_experiment_id=ophys_experiment_id)

        monitor_delay = calculate_monitor_delay(
            sync_file=sync_file, equipment=meta.behavior_metadata.equipment)

        stimulus_timestamps = StimulusTimestamps.from_sync_file(
            sync_file=sync_file,
            monitor_delay=monitor_delay)

        behavior_session_id = BehaviorSessionId.from_lims(
            db=lims_db, ophys_experiment_id=ophys_experiment_id)
        date_of_acquisition = DateOfAcquisitionOphys.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        behavior_session = BehaviorSession.from_lims(
            lims_db=lims_db,
            behavior_session_id=behavior_session_id.value,
            stimulus_timestamps=stimulus_timestamps,
            monitor_delay=monitor_delay,
            date_of_acquisition=date_of_acquisition
        )
        if is_multiplane_session:
            ophys_timestamps = OphysTimestampsMultiplane.from_sync_file(
                sync_file=sync_file,
                group_count=meta.ophys_metadata.imaging_plane_group_count,
                plane_group=meta.ophys_metadata.imaging_plane_group
            )
        else:
            ophys_timestamps = OphysTimestamps.from_sync_file(
                sync_file=sync_file)

        projections = Projections.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)
        cell_specimens = CellSpecimens.from_lims(
            ophys_experiment_id=ophys_experiment_id, lims_db=lims_db,
            ophys_timestamps=ophys_timestamps,
            segmentation_mask_image_spacing=projections.max_projection.spacing,
            events_params=EventsParams(
                filter_scale_seconds=events_filter_scale_seconds,
                filter_n_time_steps=events_filter_n_time_steps),
            exclude_invalid_rois=exclude_invalid_rois
        )
        motion_correction = _get_motion_correction()
        if skip_eye_tracking:
            eye_tracking_table = None
            eye_tracking_rig_geometry = None
        else:
            eye_tracking_table = _get_eye_tracking_table(sync_file=sync_file)
            eye_tracking_rig_geometry = EyeTrackingRigGeometry.from_lims(
                ophys_experiment_id=ophys_experiment_id, lims_db=lims_db)

        return BehaviorOphysExperiment(
            behavior_session=behavior_session,
            cell_specimens=cell_specimens,
            ophys_timestamps=ophys_timestamps,
            metadata=meta,
            projections=projections,
            motion_correction=motion_correction,
            eye_tracking_table=eye_tracking_table,
            eye_tracking_rig_geometry=eye_tracking_rig_geometry,
            date_of_acquisition=date_of_acquisition
        )

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile,
                 eye_tracking_z_threshold: float = 3.0,
                 eye_tracking_dilation_frames: int = 2,
                 events_filter_scale_seconds: float = 2.0/31.0,
                 events_filter_n_time_steps: int = 20,
                 exclude_invalid_rois=True
                 ) -> "BehaviorOphysExperiment":
        """

        Parameters
        ----------
        nwbfile
        eye_tracking_z_threshold : float, optional
            The z-threshold when determining which frames likely contain
            outliers for eye or pupil areas. Influences which frames
            are considered 'likely blinks'. By default 3.0
        eye_tracking_dilation_frames : int, optional
            Determines the number of adjacent frames that will be marked
            as 'likely_blink' when performing blink detection for
            `eye_tracking` data, by default 2
        events_filter_scale_seconds : float, optional
            Stdev of halfnorm distribution used to convolve ophys events with
            a 1d causal half-gaussian filter to smooth it for visualization,
            in seconds (by default 2.0/31.0; this value has been found to
            perform well on Allen Institute data across multiple platforms).
        events_filter_n_time_steps : int, optional
            Number of time steps to use for convolution of ophys events
        exclude_invalid_rois
            Whether to exclude invalid rois
        """
        def _is_multi_plane_session():
            imaging_plane_group_meta = ImagingPlaneGroup.from_nwb(
                nwbfile=nwbfile)
            return cls._is_multi_plane_session(
                imaging_plane_group_meta=imaging_plane_group_meta)

        behavior_session = BehaviorSession.from_nwb(nwbfile=nwbfile)
        projections = Projections.from_nwb(nwbfile=nwbfile)
        cell_specimens = CellSpecimens.from_nwb(
            nwbfile=nwbfile,
            segmentation_mask_image_spacing=projections.max_projection.spacing,
            events_params=EventsParams(
                filter_scale_seconds=events_filter_scale_seconds,
                filter_n_time_steps=events_filter_n_time_steps
            ),
            exclude_invalid_rois=exclude_invalid_rois
        )
        eye_tracking_rig_geometry = EyeTrackingRigGeometry.from_nwb(
            nwbfile=nwbfile)
        eye_tracking_table = EyeTrackingTable.from_nwb(
            nwbfile=nwbfile, z_threshold=eye_tracking_z_threshold,
            dilation_frames=eye_tracking_dilation_frames)
        motion_correction = MotionCorrection.from_nwb(nwbfile=nwbfile)
        is_multiplane_session = _is_multi_plane_session()
        metadata = BehaviorOphysMetadata.from_nwb(
            nwbfile=nwbfile, is_multiplane=is_multiplane_session)
        if is_multiplane_session:
            ophys_timestamps = OphysTimestampsMultiplane.from_nwb(
                nwbfile=nwbfile)
        else:
            ophys_timestamps = OphysTimestamps.from_nwb(nwbfile=nwbfile)
        date_of_acquisition = DateOfAcquisitionOphys.from_nwb(nwbfile=nwbfile)

        return BehaviorOphysExperiment(
            behavior_session=behavior_session,
            cell_specimens=cell_specimens,
            eye_tracking_rig_geometry=eye_tracking_rig_geometry,
            eye_tracking_table=eye_tracking_table,
            motion_correction=motion_correction,
            metadata=metadata,
            ophys_timestamps=ophys_timestamps,
            projections=projections,
            date_of_acquisition=date_of_acquisition
        )

    @classmethod
    def from_json(cls,
                  session_data: dict,
                  eye_tracking_z_threshold: float = 3.0,
                  eye_tracking_dilation_frames: int = 2,
                  events_filter_scale_seconds: float = 2.0/31.0,
                  events_filter_n_time_steps: int = 20,
                  exclude_invalid_rois=True,
                  skip_eye_tracking=False) -> \
            "BehaviorOphysExperiment":
        """

        Parameters
        ----------
        session_data
        eye_tracking_z_threshold
            See `BehaviorOphysExperiment.from_nwb`
        eye_tracking_dilation_frames
            See `BehaviorOphysExperiment.from_nwb`
        events_filter_scale_seconds
            See `BehaviorOphysExperiment.from_nwb`
        events_filter_n_time_steps
            See `BehaviorOphysExperiment.from_nwb`
        exclude_invalid_rois
            Whether to exclude invalid rois
        skip_eye_tracking
            Used to skip returning eye tracking data

        """
        def _is_multi_plane_session():
            imaging_plane_group_meta = ImagingPlaneGroup.from_json(
                dict_repr=session_data)
            return cls._is_multi_plane_session(
                imaging_plane_group_meta=imaging_plane_group_meta)

        def _get_motion_correction():
            rigid_motion_transform_file = RigidMotionTransformFile.from_json(
                dict_repr=session_data)
            return MotionCorrection.from_data_file(
                rigid_motion_transform_file=rigid_motion_transform_file)

        def _get_eye_tracking_table(sync_file: SyncFile):
            eye_tracking_file = EyeTrackingFile.from_json(
                dict_repr=session_data)
            eye_tracking_table = EyeTrackingTable.from_data_file(
                data_file=eye_tracking_file,
                sync_file=sync_file,
                z_threshold=eye_tracking_z_threshold,
                dilation_frames=eye_tracking_dilation_frames
            )
            return eye_tracking_table

        sync_file = SyncFile.from_json(dict_repr=session_data)
        is_multiplane_session = _is_multi_plane_session()
        meta = BehaviorOphysMetadata.from_json(
            dict_repr=session_data, is_multiplane=is_multiplane_session)
        monitor_delay = calculate_monitor_delay(
            sync_file=sync_file, equipment=meta.behavior_metadata.equipment)
        behavior_session = BehaviorSession.from_json(
            session_data=session_data,
            monitor_delay=monitor_delay
        )

        if is_multiplane_session:
            ophys_timestamps = OphysTimestampsMultiplane.from_sync_file(
                sync_file=sync_file,
                group_count=meta.ophys_metadata.imaging_plane_group_count,
                plane_group=meta.ophys_metadata.imaging_plane_group
            )
        else:
            ophys_timestamps = OphysTimestamps.from_sync_file(
                sync_file=sync_file)

        projections = Projections.from_json(dict_repr=session_data)
        cell_specimens = CellSpecimens.from_json(
            dict_repr=session_data,
            ophys_timestamps=ophys_timestamps,
            segmentation_mask_image_spacing=projections.max_projection.spacing,
            events_params=EventsParams(
                filter_scale_seconds=events_filter_scale_seconds,
                filter_n_time_steps=events_filter_n_time_steps),
            exclude_invalid_rois=exclude_invalid_rois
        )
        motion_correction = _get_motion_correction()
        if skip_eye_tracking:
            eye_tracking_table = None
            eye_tracking_rig_geometry = None
        else:
            eye_tracking_table = _get_eye_tracking_table(sync_file=sync_file)
            eye_tracking_rig_geometry = EyeTrackingRigGeometry.from_json(
                dict_repr=session_data)

        return BehaviorOphysExperiment(
            behavior_session=behavior_session,
            cell_specimens=cell_specimens,
            ophys_timestamps=ophys_timestamps,
            metadata=meta,
            projections=projections,
            motion_correction=motion_correction,
            eye_tracking_table=eye_tracking_table,
            eye_tracking_rig_geometry=eye_tracking_rig_geometry,
            date_of_acquisition=behavior_session._date_of_acquisition
        )

    # ========================= 'get' methods ==========================

    def get_segmentation_mask_image(self) -> Image:
        """a 2D binary image of all valid cell masks

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to segmentation_mask image data and
            metadata
        """
        return self._cell_specimens.segmentation_mask_image

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

    # ====================== properties ========================

    @property
    def ophys_experiment_id(self) -> int:
        """Unique identifier for this experimental session.
        :rtype: int
        """
        return self._metadata.ophys_metadata.ophys_experiment_id

    @property
    def ophys_session_id(self) -> int:
        """Unique identifier for this ophys session.
        :rtype: int
        """
        return self._metadata.ophys_metadata.ophys_session_id

    @property
    def metadata(self):
        behavior_meta = super()._get_metadata(
            behavior_metadata=self._metadata.behavior_metadata)
        ophys_meta = {
            'indicator': self._cell_specimens.meta.imaging_plane.indicator,
            'emission_lambda': self._cell_specimens.meta.emission_lambda,
            'excitation_lambda':
                self._cell_specimens.meta.imaging_plane.excitation_lambda,
            'experiment_container_id':
                self._metadata.ophys_metadata.experiment_container_id,
            'field_of_view_height':
                self._metadata.ophys_metadata.field_of_view_shape.height,
            'field_of_view_width':
                self._metadata.ophys_metadata.field_of_view_shape.width,
            'imaging_depth': self._metadata.ophys_metadata.imaging_depth,
            'imaging_plane_group':
                self._metadata.ophys_metadata.imaging_plane_group
                if isinstance(self._metadata.ophys_metadata,
                              MultiplaneMetadata) else None,
            'imaging_plane_group_count':
                self._metadata.ophys_metadata.imaging_plane_group_count
                if isinstance(self._metadata.ophys_metadata,
                              MultiplaneMetadata) else 0,
            'ophys_experiment_id':
                self._metadata.ophys_metadata.ophys_experiment_id,
            'ophys_frame_rate':
                self._cell_specimens.meta.imaging_plane.ophys_frame_rate,
            'ophys_session_id': self._metadata.ophys_metadata.ophys_session_id,
            'project_code': self._metadata.ophys_metadata.project_code,
            'targeted_structure':
                self._cell_specimens.meta.imaging_plane.targeted_structure
        }
        return {
            **behavior_meta,
            **ophys_meta
        }

    @property
    def max_projection(self) -> Image:
        """2D max projection image.
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        return self._projections.max_projection

    @property
    def average_projection(self) -> Image:
        """2D image of the microscope field of view, averaged across the
        experiment
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        return self._projections.avg_projection

    @property
    def ophys_timestamps(self) -> np.ndarray:
        """Timestamps associated with frames captured by the microscope
        :rtype: numpy.ndarray
        """
        return self._ophys_timestamps.value

    @property
    def dff_traces(self) -> pd.DataFrame:
        """traces of change in fluoescence / fluorescence

        Returns
        -------
        pd.DataFrame
            dataframe of traces of dff
            (change in fluorescence / fluorescence)

            dataframe columns:
                cell_specimen_id [index]: (int)
                    unified id of segmented cell across experiments
                    assigned after cell matching
                cell_roi_id: (int)
                    experiment specific id of segmented roi,
                    assigned before cell matching
                dff: (list of float)
                    fluorescence fractional values relative to baseline
                    (arbitrary units)

        """
        return self._cell_specimens.dff_traces

    @property
    def events(self) -> pd.DataFrame:
        """A dataframe containing spiking events in traces derived
        from the two photon movies, organized by cell specimen id.
        For more information on event detection processing
        please see the event detection portion of the white paper.

        Returns
        -------
        pd.DataFrame
            cell_specimen_id [index]: (int)
                unified id of segmented cell across experiments
                (assigned after cell matching)
            cell_roi_id: (int)
                experiment specific id of segmented roi (assigned
                before cell matching)
            events: (np.array of float)
                event trace where events correspond to the rise time
                of a calcium transient in the dF/F trace, with a
                magnitude roughly proportional the magnitude of the
                increase in dF/F.
            filtered_events: (np.array of float)
                Events array with a 1d causal half-gaussian filter to
                smooth it for visualization. Uses a halfnorm
                distribution as weights to the filter
            lambdas: (float64)
                regularization value selected to make the minimum
                event size be close to N * noise_std
            noise_stds: (float64)
                estimated noise standard deviation for the events trace

        """
        return self._cell_specimens.events

    @property
    def cell_specimen_table(self) -> pd.DataFrame:
        """Cell information organized into a dataframe. Table only
        contains roi_valid = True entries, as invalid ROIs/ non cell
        segmented objects have been filtered out

        Returns
        -------
        pd.DataFrame
            dataframe columns:
                cell_specimen_id [index]: (int)
                    unified id of segmented cell across experiments
                    (assigned after cell matching)
                cell_roi_id: (int)
                    experiment specific id of segmented roi
                    (assigned before cell matching)
                height: (int)
                    height of ROI/cell in pixels
                mask_image_plane: (int)
                    which image plane an ROI resides on. Overlapping
                    ROIs are stored on different mask image planes
                max_corretion_down: (float)
                    max motion correction in down direction in pixels
                max_correction_left: (float)
                    max motion correction in left direction in pixels
                max_correction_right: (float)
                    max motion correction in right direction in pixels
                max_correction_up: (float)
                    max motion correction in up direction in pixels
                roi_mask: (array of bool)
                    an image array that displays the location of the
                    roi mask in the field of view
                valid_roi: (bool)
                    indicates if cell classification found the segmented
                    ROI to be a cell or not (True = cell, False = not cell).
                width: (int)
                    width of ROI in pixels
                x: (float)
                    x position of ROI in field of view in pixels (top
                    left corner)
                y: (float)
                    y position of ROI in field of view in pixels (top
                    left corner)
        """
        return self._cell_specimens.table

    @property
    def corrected_fluorescence_traces(self) -> pd.DataFrame:
        """Corrected fluorescence traces which are neuropil corrected
        and demixed. Sampling rate can be found in metadata
        ‘ophys_frame_rate’

        Returns
        -------
        pd.DataFrame
            Dataframe that contains the corrected fluorescence traces
            for all valid cells.

            dataframe columns:
                cell_specimen_id [index]: (int)
                    unified id of segmented cell across experiments
                    (assigned after cell matching)
                cell_roi_id: (int)
                    experiment specific id of segmented roi
                    (assigned before cell matching)
                corrected_fluorescence: (list of float)
                    fluorescence values (arbitrary units)

        """
        return self._cell_specimens.corrected_fluorescence_traces

    @property
    def motion_correction(self) -> pd.DataFrame:
        """a dataframe containing the x and y offsets applied during
        motion correction

        Returns
        -------
        pd.DataFrame
            dataframe columns:
                x: (int)
                    frame shift along x axis
                y: (int)
                    frame shift along y axis
        """
        return self._motion_correction.value

    @property
    def segmentation_mask_image(self) -> Image:
        """A 2d binary image of all valid cell masks
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        return self._cell_specimens.segmentation_mask_image

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
          fit failed OR eye fit failed OR an outlier fit was identified on the
          pupil or eye fit.
        - The pupil_area, cr_area, eye_area columns are set to NaN wherever
          'likely_blink' == True.
        - The pupil_area_raw, cr_area_raw, eye_area_raw columns contains all
          pupil fit values (including where 'likely_blink' == True).
        - All ellipse fits are derived from tracking points that were output by
          a DeepLabCut model that was trained on hand-annotated data from a
          subset of imaging sessions on optical physiology rigs.
        - Raw DeepLabCut tracking points are not publicly available.

        :rtype: pandas.DataFrame
        """
        return self._eye_tracking.value

    @property
    def eye_tracking_rig_geometry(self) -> dict:
        """the eye tracking equipment geometry associate with a
        given ophys experiment session.

        Returns
        -------
        dict
            dictionary with the following keys:
                camera_eye_position_mm (array of float)
                camera_rotation_deg (array of float)
                equipment (string)
                led_position (array of float)
                monitor_position_mm (array of float)
                monitor_rotation_deg (array of float)
        """
        if self._eye_tracking_rig_geometry is None:
            return dict()
        return self._eye_tracking_rig_geometry.to_dict()['rig_geometry']

    @property
    def roi_masks(self) -> pd.DataFrame:
        return self.cell_specimen_table[['cell_roi_id', 'roi_mask']]

    def _get_identifier(self) -> str:
        return str(self.ophys_experiment_id)

    @staticmethod
    def _is_multi_plane_session(
            imaging_plane_group_meta: ImagingPlaneGroup) -> bool:
        """Returns whether this experiment is part of a multiplane session"""
        return imaging_plane_group_meta is not None and \
            imaging_plane_group_meta.plane_group_count > 1

    def _get_session_type(self) -> str:
        return self._metadata.behavior_metadata.session_type

    @staticmethod
    def _get_keywords():
        """Keywords for NWB file"""
        return ["2-photon", "calcium imaging", "visual cortex",
                "behavior", "task"]
