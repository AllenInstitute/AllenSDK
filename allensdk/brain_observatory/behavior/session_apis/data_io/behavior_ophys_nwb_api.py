import datetime
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pynwb
import pytz
import SimpleITK as sitk
from hdmf.backends.hdf5 import H5DataIO

from pynwb import NWBHDF5IO, NWBFile

from allensdk.brain_observatory.behavior.metadata.behavior_ophys_metadata \
    import BehaviorOphysMetadata
from allensdk.brain_observatory.behavior.event_detection import \
    filter_events_array
import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.metadata.behavior_metadata import (
    get_expt_description
)
from allensdk.brain_observatory.behavior.session_apis.abcs.session_base. \
    behavior_ophys_base import BehaviorOphysBase
from allensdk.brain_observatory.behavior.schemas import (
    BehaviorTaskParametersSchema, OphysEyeTrackingRigMetadataSchema)
from allensdk.brain_observatory.behavior.trials_processing import (
    TRIAL_COLUMN_DESCRIPTION_DICT
)
from allensdk.brain_observatory.nwb import TimeSeries
from allensdk.brain_observatory.nwb.eye_tracking.ndx_ellipse_eye_tracking import (  # noqa: E501
    EllipseEyeTracking, EllipseSeries)
from allensdk.brain_observatory.behavior.write_nwb.extensions \
    .event_detection.ndx_ophys_events import OphysEventDetection
from allensdk.brain_observatory.nwb.metadata import load_pynwb_extension
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorNwbApi
)
from allensdk.brain_observatory.nwb.nwb_utils import set_omitted_stop_time
from allensdk.brain_observatory.behavior.eye_tracking_processing import (
    determine_outliers, determine_likely_blinks
)

load_pynwb_extension(BehaviorTaskParametersSchema, 'ndx-aibs-behavior-ophys')


class BehaviorOphysNwbApi(BehaviorNwbApi, BehaviorOphysBase):
    """A data fetching class that serves as an API for fetching 'raw'
    data from an NWB file that is both necessary and sufficient for filling
    a 'BehaviorOphysExperiment'.
    """

    def __init__(self, *args, **kwargs):
        self.filter_invalid_rois = kwargs.pop("filter_invalid_rois", False)
        super().__init__(*args, **kwargs)

    def save(self, session_object):
        # Cannot type session_object due to a circular dependency
        # TODO fix circular dependency and add type

        session_metadata: BehaviorOphysMetadata = \
            session_object.api.get_metadata()

        session_type = session_metadata.session_type

        nwbfile = NWBFile(
            session_description=session_type,
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_metadata.date_of_acquisition,
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=["2-photon", "calcium imaging", "visual cortex",
                      "behavior", "task"],
            experiment_description=get_expt_description(session_type)
        )

        # Add stimulus_timestamps to NWB in-memory object:
        nwb.add_stimulus_timestamps(nwbfile,
                                    session_object.stimulus_timestamps)

        # Add running acquisition ('dx', 'v_sig', 'v_in') data to NWB
        # This data should be saved to NWB but not accessible directly from
        # Sessions
        nwb.add_running_acquisition_to_nwbfile(
            nwbfile,
            session_object.api.get_running_acquisition_df())

        # Add running data to NWB in-memory object:
        nwb.add_running_speed_to_nwbfile(nwbfile,
                                         session_object.running_speed,
                                         name="speed",
                                         from_dataframe=True)
        nwb.add_running_speed_to_nwbfile(nwbfile,
                                         session_object.raw_running_speed,
                                         name="speed_unfiltered",
                                         from_dataframe=True)

        # Add stimulus template data to NWB in-memory object:
        # Use the semi-private _stimulus_templates attribute because it is
        # a StimulusTemplate object. The public stimulus_templates property
        # of the session_object returns a DataFrame.
        session_stimulus_templates = session_object._stimulus_templates
        self._add_stimulus_templates(
            nwbfile=nwbfile,
            stimulus_templates=session_stimulus_templates,
            stimulus_presentations=session_object.stimulus_presentations)

        # search for omitted rows and add stop_time before writing to NWB file
        set_omitted_stop_time(
            stimulus_table=session_object.stimulus_presentations)

        # Add stimulus presentations data to NWB in-memory object:
        nwb.add_stimulus_presentations(nwbfile,
                                       session_object.stimulus_presentations)

        # Add trials data to NWB in-memory object:
        nwb.add_trials(nwbfile, session_object.trials,
                       TRIAL_COLUMN_DESCRIPTION_DICT)

        # Add licks data to NWB in-memory object:
        if len(session_object.licks) > 0:
            nwb.add_licks(nwbfile, session_object.licks)

        # Add rewards data to NWB in-memory object:
        if len(session_object.rewards) > 0:
            nwb.add_rewards(nwbfile, session_object.rewards)

        # Add max_projection image data to NWB in-memory object:
        nwb.add_max_projection(nwbfile, session_object.max_projection)

        # Add average_image image data to NWB in-memory object:
        nwb.add_average_image(nwbfile, session_object.average_projection)

        # Add segmentation_mask_image image data to NWB in-memory object:
        nwb.add_segmentation_mask_image(nwbfile,
                                        session_object.segmentation_mask_image)

        # Add metadata to NWB in-memory object:
        nwb.add_metadata(nwbfile, session_object.metadata,
                         behavior_only=False)

        # Add task parameters to NWB in-memory object:
        nwb.add_task_parameters(nwbfile, session_object.task_parameters)

        # Add roi metrics to NWB in-memory object:
        nwb.add_cell_specimen_table(nwbfile,
                                    session_object.cell_specimen_table,
                                    session_object.metadata)

        # Add dff to NWB in-memory object:
        nwb.add_dff_traces(nwbfile, session_object.dff_traces,
                           session_object.ophys_timestamps)

        # Add corrected_fluorescence to NWB in-memory object:
        nwb.add_corrected_fluorescence_traces(
            nwbfile,
            session_object.corrected_fluorescence_traces)

        # Add motion correction to NWB in-memory object:
        nwb.add_motion_correction(nwbfile, session_object.motion_correction)

        # Add eye tracking and rig geometry to NWB in-memory object
        # if eye_tracking data exists.
        if session_object.eye_tracking is not None:
            self.add_eye_tracking_data_to_nwb(
                nwbfile,
                session_object.eye_tracking,
                session_object.eye_tracking_rig_geometry)

        # Add events
        self.add_events(nwbfile=nwbfile, events=session_object.events)

        # Write the file:
        with NWBHDF5IO(self.path, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_behavior_session_id(self) -> int:
        return self.get_metadata()['behavior_session_id']

    def get_ophys_session_id(self) -> int:
        return self.get_metadata()['ophys_session_id']

    def get_ophys_experiment_id(self) -> int:
        return int(self.nwbfile.identifier)

    def get_eye_tracking(self,
                         z_threshold: float = 3.0,
                         dilation_frames: int = 2) -> Optional[pd.DataFrame]:
        """
        Gets corneal, eye, and pupil ellipse fit data

        Parameters
        ----------
        z_threshold : float, optional
            The z-threshold when determining which frames likely contain
            outliers for eye or pupil areas. Influences which frames
            are considered 'likely blinks'. By default 3.0
        dilation_frames : int, optional
             Determines the number of additional adjacent frames to mark as
            'likely_blink', by default 2.

        Returns
        -------
        pd.DataFrame
            *_area
            *_center_x
            *_center_y
            *_height
            *_phi
            *_width
            likely_blink
        where "*" can be "corneal", "pupil" or "eye"
        or None if no eye tracking data
        Note: `pupil_area` is set to NaN where `likely_blink` == True
              use `pupil_area_raw` column to access unfiltered pupil data
        """
        try:
            eye_tracking_acquisition = self.nwbfile.acquisition['EyeTracking']
        except KeyError as e:
            warnings.warn("This ophys session "
                          f"'{int(self.nwbfile.identifier)}' has no eye "
                          f"tracking data. (NWB error: {e})")
            return None

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

            "eye_center_x": eye_tracking.data[:, 0],
            "eye_center_y": eye_tracking.data[:, 1],
            "eye_area_raw": eye_tracking.area_raw[:],
            "eye_height": eye_tracking.height[:],
            "eye_width": eye_tracking.width[:],
            "eye_phi": eye_tracking.angle[:],

            "pupil_center_x": pupil_tracking.data[:, 0],
            "pupil_center_y": pupil_tracking.data[:, 1],
            "pupil_area_raw": pupil_tracking.area_raw[:],
            "pupil_height": pupil_tracking.height[:],
            "pupil_width": pupil_tracking.width[:],
            "pupil_phi": pupil_tracking.angle[:],

            "cr_center_x": corneal_reflection_tracking.data[:, 0],
            "cr_center_y": corneal_reflection_tracking.data[:, 1],
            "cr_area_raw": corneal_reflection_tracking.area_raw[:],
            "cr_height": corneal_reflection_tracking.height[:],
            "cr_width": corneal_reflection_tracking.width[:],
            "cr_phi": corneal_reflection_tracking.angle[:],
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
        eye_tracking_data.at[likely_blinks, "eye_area"] = np.nan
        eye_tracking_data.at[likely_blinks, "pupil_area"] = np.nan
        eye_tracking_data.at[likely_blinks, "cr_area"] = np.nan

        return eye_tracking_data

    def get_eye_tracking_rig_geometry(self) -> Optional[dict]:
        try:
            et_mod = \
                self.nwbfile.get_processing_module("eye_tracking_rig_metadata")
        except KeyError as e:
            warnings.warn("This ophys session "
                          f"'{int(self.nwbfile.identifier)}' has no eye "
                          f"tracking rig metadata. (NWB error: {e})")
            return None

        meta = et_mod.get_data_interface("eye_tracking_rig_metadata")

        monitor_position = meta.monitor_position[:]
        monitor_position = (monitor_position.tolist()
                            if isinstance(monitor_position, np.ndarray)
                            else monitor_position)

        monitor_rotation = meta.monitor_rotation[:]
        monitor_rotation = (monitor_rotation.tolist()
                            if isinstance(monitor_rotation, np.ndarray)
                            else monitor_rotation)

        camera_position = meta.camera_position[:]
        camera_position = (camera_position.tolist()
                           if isinstance(camera_position, np.ndarray)
                           else camera_position)

        camera_rotation = meta.camera_rotation[:]
        camera_rotation = (camera_rotation.tolist()
                           if isinstance(camera_rotation, np.ndarray)
                           else camera_rotation)

        led_position = meta.led_position[:]
        led_position = (led_position.tolist()
                        if isinstance(led_position, np.ndarray)
                        else led_position)

        rig_geometry = {
            f"monitor_position_{meta.monitor_position__unit_of_measurement}":
                monitor_position,
            f"camera_position_{meta.camera_position__unit_of_measurement}":
                camera_position,
            "led_position": led_position,
            f"monitor_rotation_{meta.monitor_rotation__unit_of_measurement}":
                monitor_rotation,
            f"camera_rotation_{meta.camera_rotation__unit_of_measurement}":
                camera_rotation,
            "equipment": meta.equipment
        }

        return rig_geometry

    def get_ophys_timestamps(self) -> np.ndarray:
        return self.nwbfile.processing[
                   'ophys'].get_data_interface('dff').roi_response_series[
                   'traces'].timestamps[:]

    def get_max_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('max_projection', 'ophys', image_api=image_api)

    def get_average_projection(self, image_api=None) -> sitk.Image:
        return self.get_image('average_image', 'ophys', image_api=image_api)

    def get_segmentation_mask_image(self, image_api=None) -> sitk.Image:
        return self.get_image('segmentation_mask_image',
                              'ophys', image_api=image_api)

    def get_metadata(self) -> dict:
        data = super().get_metadata()

        # Add pyNWB OpticalChannel and ImagingPlane metadata to behavior ophys
        # session metadata
        try:
            ophys_module = self.nwbfile.processing['ophys']
        except KeyError:
            warnings.warn("Could not locate 'ophys' module in "
                          "NWB file. The following metadata fields will be "
                          "missing: 'ophys_frame_rate', 'indicator', "
                          "'targeted_structure', 'excitation_lambda', "
                          "'emission_lambda'")
        else:
            image_seg = ophys_module.data_interfaces['image_segmentation']
            imaging_plane = image_seg.plane_segmentations[
                'cell_specimen_table'].imaging_plane
            optical_channel = imaging_plane.optical_channel[0]

            data['ophys_frame_rate'] = imaging_plane.imaging_rate
            data['indicator'] = imaging_plane.indicator
            data['targeted_structure'] = imaging_plane.location
            data['excitation_lambda'] = imaging_plane.excitation_lambda
            data['emission_lambda'] = optical_channel.emission_lambda

        # Because nwb can't store imaging_plane_group as None
        nwb_imaging_plane_group = data['imaging_plane_group']
        if nwb_imaging_plane_group == -1:
            data["imaging_plane_group"] = None
        else:
            data["imaging_plane_group"] = nwb_imaging_plane_group

        return data

    def get_cell_specimen_table(self) -> pd.DataFrame:
        # NOTE: ROI masks are stored in full frame width and height arrays
        df = self.nwbfile.processing[
            'ophys'].data_interfaces[
            'image_segmentation'].plane_segmentations[
            'cell_specimen_table'].to_dataframe()

        # Because pynwb stores this field as "image_mask", it is renamed here
        df = df.rename(columns={'image_mask': 'roi_mask'})

        df.index.rename('cell_roi_id', inplace=True)
        df['cell_specimen_id'] = [None if csid == -1 else csid
                                  for csid in df['cell_specimen_id'].values]

        df.reset_index(inplace=True)
        df.set_index('cell_specimen_id', inplace=True)

        if self.filter_invalid_rois:
            df = df[df["valid_roi"]]

        return df

    def get_dff_traces(self) -> pd.DataFrame:
        dff_nwb = self.nwbfile.processing[
            'ophys'].data_interfaces['dff'].roi_response_series['traces']
        # dff traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        dff_traces = dff_nwb.data[:].T
        number_of_cells, number_of_dff_frames = dff_traces.shape
        num_of_timestamps = len(self.get_ophys_timestamps())
        assert num_of_timestamps == number_of_dff_frames

        df = pd.DataFrame({'dff': dff_traces.tolist()},
                          index=pd.Index(data=dff_nwb.rois.table.id[:],
                                         name='cell_roi_id'))
        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[['cell_roi_id']].join(df, on='cell_roi_id')
        return df

    def get_corrected_fluorescence_traces(self) -> pd.DataFrame:
        corr_fluorescence_nwb = self.nwbfile.processing[
            'ophys'].data_interfaces[
            'corrected_fluorescence'].roi_response_series['traces']
        # f traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        f_traces = corr_fluorescence_nwb.data[:].T
        df = pd.DataFrame({'corrected_fluorescence': f_traces.tolist()},
                          index=pd.Index(
                              data=corr_fluorescence_nwb.rois.table.id[:],
                              name='cell_roi_id'))

        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[['cell_roi_id']].join(df, on='cell_roi_id')
        return df

    def get_motion_correction(self) -> pd.DataFrame:
        ophys_module = self.nwbfile.processing['ophys']

        motion_correction_data = {}
        motion_correction_data['x'] = ophys_module.get_data_interface(
            'ophys_motion_correction_x').data[:]
        motion_correction_data['y'] = ophys_module.get_data_interface(
            'ophys_motion_correction_y').data[:]

        return pd.DataFrame(motion_correction_data)

    def add_eye_tracking_data_to_nwb(self, nwbfile: NWBFile,
                                     eye_tracking_df: pd.DataFrame,
                                     eye_tracking_rig_geometry: Optional[dict]
                                     ) -> NWBFile:
        # 1. Add rig geometry
        if eye_tracking_rig_geometry:
            self.add_eye_tracking_rig_geometry_data_to_nwbfile(
                nwbfile=nwbfile,
                eye_tracking_rig_geometry=eye_tracking_rig_geometry)

        # 2. Add eye tracking
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

    @staticmethod
    def add_eye_tracking_rig_geometry_data_to_nwbfile(
            nwbfile: NWBFile, eye_tracking_rig_geometry: dict) -> NWBFile:
        """ Rig geometry dict should consist of the following fields:
        monitor_position_mm: [x, y, z]
        monitor_rotation_deg: [x, y, z]
        camera_position_mm: [x, y, z]
        camera_rotation_deg: [x, y, z]
        led_position: [x, y, z]
        equipment: A string describing rig
        """
        eye_tracking_rig_mod = pynwb.ProcessingModule(
            name='eye_tracking_rig_metadata',
            description='Eye tracking rig metadata module')

        ophys_eye_tracking_rig_metadata = load_pynwb_extension(
            OphysEyeTrackingRigMetadataSchema, 'ndx-aibs-behavior-ophys')

        rig_metadata = ophys_eye_tracking_rig_metadata(
            name="eye_tracking_rig_metadata",
            equipment=eye_tracking_rig_geometry['equipment'],
            monitor_position=eye_tracking_rig_geometry['monitor_position_mm'],
            monitor_position__unit_of_measurement="mm",
            camera_position=eye_tracking_rig_geometry['camera_position_mm'],
            camera_position__unit_of_measurement="mm",
            led_position=eye_tracking_rig_geometry['led_position'],
            led_position__unit_of_measurement="mm",
            monitor_rotation=eye_tracking_rig_geometry['monitor_rotation_deg'],
            monitor_rotation__unit_of_measurement="deg",
            camera_rotation=eye_tracking_rig_geometry['camera_rotation_deg'],
            camera_rotation__unit_of_measurement="deg"
        )

        eye_tracking_rig_mod.add_data_interface(rig_metadata)
        nwbfile.add_processing_module(eye_tracking_rig_mod)

        return nwbfile

    def get_events(self, filter_scale: float = 2,
                   filter_n_time_steps: int = 20) -> pd.DataFrame:
        """
        Parameters
        ----------
        filter_scale: float
            See filter_events_array for description
        filter_n_time_steps: int
            See filter_events_array for description

        Returns
        -------
        Events dataframe:
            columns:
            events: np.array
            lambda: float
            noise_std: float
            cell_roi_id: int

            index:
            cell_specimen_id: int

        """
        event_detection = self.nwbfile.processing['ophys']['event_detection']
        # NOTE: The rois with events are stored in event detection
        partial_cell_specimen_table = event_detection.rois.to_dataframe()

        events = event_detection.data[:]

        # events stored time x roi. Change back to roi x time
        events = events.T

        filtered_events = filter_events_array(
            arr=events, scale=filter_scale, n_time_steps=filter_n_time_steps)

        # Convert to list to that it can be stored in a single column
        events = [x for x in events]
        filtered_events = [x for x in filtered_events]

        return pd.DataFrame({
            'cell_roi_id': partial_cell_specimen_table.index,
            'events': events,
            'filtered_events': filtered_events,
            'lambda': event_detection.lambdas[:],
            'noise_std': event_detection.noise_stds[:]
        }, index=pd.Index(partial_cell_specimen_table['cell_specimen_id']))

    @staticmethod
    def add_events(nwbfile: NWBFile, events: pd.DataFrame) -> NWBFile:
        """
        Adds events to NWB file from dataframe

        Returns
        -------
        NWBFile:
            The NWBFile with events added
        """
        events_data = np.vstack(events['events'])

        ophys_module = nwbfile.processing['ophys']
        dff_interface = ophys_module.data_interfaces['dff']
        traces = dff_interface.roi_response_series['traces']
        seg_interface = ophys_module.data_interfaces['image_segmentation']

        cell_specimen_table = (
            seg_interface.plane_segmentations['cell_specimen_table'])
        cell_specimen_df = cell_specimen_table.to_dataframe()
        cell_specimen_df = cell_specimen_df.set_index('cell_specimen_id')
        # We only want to store the subset of rois that have events data
        rois_with_events_indices = [cell_specimen_df.index.get_loc(label)
                                    for label in events.index]
        roi_table_region = cell_specimen_table.create_roi_table_region(
            description="Cells with detected events",
            region=rois_with_events_indices)

        events = OphysEventDetection(
            # time x rois instead of rois x time
            # store using compression since sparse
            data=H5DataIO(events_data.T, compression=True),

            lambdas=events['lambda'].values,
            noise_stds=events['noise_std'].values,
            unit='N/A',
            rois=roi_table_region,
            timestamps=traces.timestamps
        )

        ophys_module.add_data_interface(events)

        return nwbfile
