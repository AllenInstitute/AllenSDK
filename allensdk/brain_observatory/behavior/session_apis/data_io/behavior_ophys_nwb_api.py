import datetime
import uuid
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pynwb
import pytz
import SimpleITK as sitk

from pynwb import NWBHDF5IO, NWBFile

import allensdk.brain_observatory.nwb as nwb
from allensdk.brain_observatory.behavior.metadata_processing import (
    get_expt_description,
)
from allensdk.brain_observatory.behavior.session_apis.abcs import (
    BehaviorOphysBase,
)
from allensdk.brain_observatory.behavior.schemas import (
    BehaviorTaskParametersSchema,
    OphysBehaviorMetadataSchema,
    OphysEyeTrackingRigMetadataSchema,
)
from allensdk.brain_observatory.behavior.trials_processing import (
    TRIAL_COLUMN_DESCRIPTION_DICT,
)
from allensdk.brain_observatory.nwb import TimeSeries
from allensdk.brain_observatory.nwb.eye_tracking.ndx_ellipse_eye_tracking import (  # noqa: E501
    EllipseEyeTracking,
    EllipseSeries,
)
from allensdk.brain_observatory.nwb.metadata import load_pynwb_extension
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorNwbApi,
)
from allensdk.brain_observatory.nwb.nwb_utils import set_omitted_stop_time

load_pynwb_extension(OphysBehaviorMetadataSchema, "ndx-aibs-behavior-ophys")
load_pynwb_extension(BehaviorTaskParametersSchema, "ndx-aibs-behavior-ophys")


class BehaviorOphysNwbApi(BehaviorNwbApi, BehaviorOphysBase):
    """A data fetching class that serves as an API for fetching 'raw'
    data from an NWB file that is both necessary and sufficient for filling
    a 'BehaviorOphysSession'.
    """

    def __init__(self, *args, **kwargs):
        self.filter_invalid_rois = kwargs.pop("filter_invalid_rois", False)
        super().__init__(*args, **kwargs)

    def save(self, session_object):

        session_type = str(session_object.metadata["session_type"])

        nwbfile = NWBFile(
            session_description=session_type,
            identifier=str(session_object.ophys_experiment_id),
            session_start_time=session_object.metadata["experiment_datetime"],
            file_create_date=pytz.utc.localize(datetime.datetime.now()),
            institution="Allen Institute for Brain Science",
            keywords=[
                "2-photon",
                "calcium imaging",
                "visual cortex",
                "behavior",
                "task",
            ],
            experiment_description=get_expt_description(session_type),
        )

        # Add stimulus_timestamps to NWB in-memory object:
        nwb.add_stimulus_timestamps(
            nwbfile, session_object.stimulus_timestamps
        )

        # Add running data to NWB in-memory object:
        nwb.add_running_speed_to_nwbfile(
            nwbfile,
            session_object.running_speed,
            name="speed",
            from_dataframe=True,
        )
        nwb.add_running_speed_to_nwbfile(
            nwbfile,
            session_object.raw_running_speed,
            name="speed_unfiltered",
            from_dataframe=True,
        )

        # Add stimulus template data to NWB in-memory object:
        for name, image_data in session_object.stimulus_templates.items():
            nwb.add_stimulus_template(nwbfile, image_data, name)

            # Add index for this template to NWB in-memory object:
            nwb_template = nwbfile.stimulus_template[name]
            stimulus_index = session_object.stimulus_presentations[
                session_object.stimulus_presentations["image_set"]
                == nwb_template.name
            ]
            nwb.add_stimulus_index(nwbfile, stimulus_index, nwb_template)

        # search for omitted rows and add stop_time before writing to NWB file
        set_omitted_stop_time(
            stimulus_table=session_object.stimulus_presentations
        )

        # Add stimulus presentations data to NWB in-memory object:
        nwb.add_stimulus_presentations(
            nwbfile, session_object.stimulus_presentations
        )

        # Add trials data to NWB in-memory object:
        nwb.add_trials(
            nwbfile, session_object.trials, TRIAL_COLUMN_DESCRIPTION_DICT
        )

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
        nwb.add_segmentation_mask_image(
            nwbfile, session_object.segmentation_mask_image
        )

        # Add metadata to NWB in-memory object:
        nwb.add_metadata(nwbfile, session_object.metadata, behavior_only=False)

        # Add task parameters to NWB in-memory object:
        nwb.add_task_parameters(nwbfile, session_object.task_parameters)

        # Add roi metrics to NWB in-memory object:
        nwb.add_cell_specimen_table(
            nwbfile,
            session_object.cell_specimen_table,
            session_object.metadata,
        )

        # Add dff to NWB in-memory object:
        nwb.add_dff_traces(
            nwbfile, session_object.dff_traces, session_object.ophys_timestamps
        )

        # Add corrected_fluorescence to NWB in-memory object:
        nwb.add_corrected_fluorescence_traces(
            nwbfile, session_object.corrected_fluorescence_traces
        )

        # Add motion correction to NWB in-memory object:
        nwb.add_motion_correction(nwbfile, session_object.motion_correction)

        # Add eye tracking, rig geometry, and gaze mapping data to NWB
        # in-memory object.
        eye_gaze_fpath = (
            session_object.api.extractor.get_eye_gaze_mapping_file_path()
        )
        self.add_eye_tracking_data_to_nwb(
            nwbfile=nwbfile,
            eye_tracking_df=session_object.eye_tracking,
            eye_tracking_rig_geometry=session_object.eye_tracking_rig_geometry,
            eye_gaze_mapping_file_path=eye_gaze_fpath,
        )

        # Write the file:
        with NWBHDF5IO(self.path, "w") as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        return nwbfile

    def get_ophys_experiment_id(self) -> int:
        return int(self.nwbfile.identifier)

    # TODO: Implement save and load of ophys_session_id to/from NWB file
    def get_ophys_session_id(self) -> int:
        raise NotImplementedError()

    def get_eye_tracking(self) -> Optional[pd.DataFrame]:
        """
        Gets corneal, eye, and pupil ellipse fit data

        Returns
        -------
        pd.DataFrame
            *_area
            *_center_x
            *_center_y
            *_height
            *_phi
            *_width
            where "*" can be "corneal", "pupil" or "eye"
            likely_blink
        or None if no eye tracking data
        """
        try:
            eye_tracking_acquisition = self.nwbfile.acquisition["EyeTracking"]
        except KeyError as e:
            warnings.warn(
                "This ophys session "
                f"'{int(self.nwbfile.identifier)}' has no eye "
                f"tracking data. (NWB error: {e})"
            )
            return None

        eye_tracking = eye_tracking_acquisition.eye_tracking
        pupil_tracking = eye_tracking_acquisition.pupil_tracking
        corneal_reflection_tracking = (
            eye_tracking_acquisition.corneal_reflection_tracking
        )

        eye_tracking_data = {
            "eye_center_x": eye_tracking.data[:, 0],
            "eye_center_y": eye_tracking.data[:, 1],
            "eye_area": eye_tracking.area[:],
            "eye_height": eye_tracking.height[:],
            "eye_width": eye_tracking.width[:],
            "eye_phi": eye_tracking.angle[:],
            "pupil_center_x": pupil_tracking.data[:, 0],
            "pupil_center_y": pupil_tracking.data[:, 1],
            "pupil_area": pupil_tracking.area[:],
            "pupil_height": pupil_tracking.height[:],
            "pupil_width": pupil_tracking.width[:],
            "pupil_phi": pupil_tracking.angle[:],
            "cr_center_x": corneal_reflection_tracking.data[:, 0],
            "cr_center_y": corneal_reflection_tracking.data[:, 1],
            "cr_area": corneal_reflection_tracking.area[:],
            "cr_height": corneal_reflection_tracking.height[:],
            "cr_width": corneal_reflection_tracking.width[:],
            "cr_phi": corneal_reflection_tracking.angle[:],
            "likely_blink": eye_tracking_acquisition.likely_blink.data[:],
            "time": eye_tracking.timestamps[:],
        }

        eye_tracking_data = pd.DataFrame(eye_tracking_data)
        eye_tracking_data.index = eye_tracking_data.index.rename("frame")
        return eye_tracking_data

    def get_eye_tracking_rig_geometry(self) -> Optional[dict]:
        try:
            et_mod = self.nwbfile.get_processing_module(
                "eye_tracking_rig_metadata"
            )
        except KeyError as e:
            warnings.warn(
                "This ophys session "
                f"'{int(self.nwbfile.identifier)}' has no eye "
                f"tracking rig metadata. (NWB error: {e})"
            )
            return None

        meta = et_mod.get_data_interface("eye_tracking_rig_metadata")

        monitor_position = meta.monitor_position[:]
        monitor_position = (
            monitor_position.tolist()
            if isinstance(monitor_position, np.ndarray)
            else monitor_position
        )

        monitor_rotation = meta.monitor_rotation[:]
        monitor_rotation = (
            monitor_rotation.tolist()
            if isinstance(monitor_rotation, np.ndarray)
            else monitor_rotation
        )

        camera_position = meta.camera_position[:]
        camera_position = (
            camera_position.tolist()
            if isinstance(camera_position, np.ndarray)
            else camera_position
        )

        camera_rotation = meta.camera_rotation[:]
        camera_rotation = (
            camera_rotation.tolist()
            if isinstance(camera_rotation, np.ndarray)
            else camera_rotation
        )

        led_position = meta.led_position[:]
        led_position = (
            led_position.tolist()
            if isinstance(led_position, np.ndarray)
            else led_position
        )

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
            "equipment": meta.equipment,
        }

        return rig_geometry

    def get_screen_gaze_data(
        self, include_filtered_data=False
    ) -> Optional[pd.DataFrame]:
        """
        Gets screen gaze data
        Parameters
        ----------
        include_filtered_data: bool
            Includes new_* data
        Returns
        -------
        pd.DataFrame
            *_eye_areas: Area of eye (in pixels^2) over time
            *_pupil_areas: Area of pupil (in pixels^2) over time
            *_screen_coordinates: y, x screen coordinates (in cm) over time
            *_screen_coordinates_spherical: y, x screen coordinates (in deg)
            over time synced_frame_timestamps: synced timestamps for video
            frames (in sec)
        or None if no eye tracking data
        """
        try:
            rgm_mod = self.nwbfile.get_processing_module("raw_gaze_mapping")
            fgm_mod = self.nwbfile.get_processing_module(
                "filtered_gaze_mapping"
            )
        except KeyError as e:
            warnings.warn(
                "This ophys session "
                f"'{int(self.nwbfile.identifier)}' has no eye "
                f"tracking data. (NWB error: {e})"
            )
            return None

        raw_eye_area_ts = rgm_mod.get_data_interface("eye_area")
        raw_pupil_area_ts = rgm_mod.get_data_interface("pupil_area")
        raw_screen_coordinates_ts = rgm_mod.get_data_interface(
            "screen_coordinates"
        )
        raw_screen_coordinates_spherical_ts = rgm_mod.get_data_interface(
            "screen_coordinates_spherical"
        )

        filtered_eye_area_ts = fgm_mod.get_data_interface("eye_area")
        filtered_pupil_area_ts = fgm_mod.get_data_interface("pupil_area")
        filtered_screen_coordinates_ts = fgm_mod.get_data_interface(
            "screen_coordinates"
        )
        filtered_screen_coordinates_spherical_ts = fgm_mod.get_data_interface(
            "screen_coordinates_spherical"
        )

        gaze_data = {
            "raw_eye_area": raw_eye_area_ts.data[:],
            "raw_pupil_area": raw_pupil_area_ts.data[:],
            "raw_screen_coordinates_x_cm":
            raw_screen_coordinates_ts.data[:, 1],
            "raw_screen_coordinates_y_cm":
            raw_screen_coordinates_ts.data[:, 0],
            "raw_screen_coordinates_spherical_x_deg":
            raw_screen_coordinates_spherical_ts.data[:, 1],
            "raw_screen_coordinates_spherical_y_deg":
            raw_screen_coordinates_spherical_ts.data[:, 0],
        }

        if include_filtered_data:
            gaze_data.update(
                {
                    "filtered_eye_area": filtered_eye_area_ts.data[:],
                    "filtered_pupil_area": filtered_pupil_area_ts.data[:],
                    "filtered_screen_coordinates_x_cm":
                    filtered_screen_coordinates_ts.data[:, 1],
                    "filtered_screen_coordinates_y_cm":
                    filtered_screen_coordinates_ts.data[:, 0],
                    "filtered_screen_coordinates_spherical_x_deg":
                    filtered_screen_coordinates_spherical_ts.data[:, 1],
                    "filtered_screen_coordinates_spherical_y_deg":
                    filtered_screen_coordinates_spherical_ts.data[:, 0],
                }
            )

        index = pd.Index(data=raw_eye_area_ts.timestamps[:], name="Time (s)")
        return pd.DataFrame(gaze_data, index=index)

    def get_ophys_timestamps(self) -> np.ndarray:
        return (
            self.nwbfile.processing["ophys"]
            .get_data_interface("dff")
            .roi_response_series["traces"]
            .timestamps[:]
        )

    def get_max_projection(self, image_api=None) -> sitk.Image:
        return self.get_image("max_projection", "ophys", image_api=image_api)

    def get_average_projection(self, image_api=None) -> sitk.Image:
        return self.get_image("average_image", "ophys", image_api=image_api)

    def get_segmentation_mask_image(self, image_api=None) -> sitk.Image:
        return self.get_image(
            "segmentation_mask_image", "ophys", image_api=image_api
        )

    def get_metadata(self) -> dict:

        metadata_nwb_obj = self.nwbfile.lab_meta_data["metadata"]
        data = OphysBehaviorMetadataSchema(
            exclude=["experiment_datetime"]
        ).dump(metadata_nwb_obj)

        # Add pyNWB Subject metadata to behavior ophys session metadata
        nwb_subject = self.nwbfile.subject
        data["LabTracks_ID"] = int(nwb_subject.subject_id)
        data["sex"] = nwb_subject.sex
        data["age"] = nwb_subject.age
        data["full_genotype"] = nwb_subject.genotype
        data["reporter_line"] = list(nwb_subject.reporter_line)
        data["driver_line"] = list(nwb_subject.driver_line)

        # Add pyNWB OpticalChannel and ImagingPlane metadata to behavior ophys
        # session metadata
        try:
            ophys_module = self.nwbfile.processing["ophys"]
        except KeyError:
            warnings.warn(
                "Could not locate 'ophys' module in "
                "NWB file. The following metadata fields will be "
                "missing: 'ophys_frame_rate', 'indicator', "
                "'targeted_structure', 'excitation_lambda', "
                "'emission_lambda'"
            )
        else:
            image_seg = ophys_module.data_interfaces["image_segmentation"]
            imaging_plane = image_seg.plane_segmentations[
                "cell_specimen_table"
            ].imaging_plane
            optical_channel = imaging_plane.optical_channel[0]

            data["ophys_frame_rate"] = imaging_plane.imaging_rate
            data["indicator"] = imaging_plane.indicator
            data["targeted_structure"] = imaging_plane.location
            data["excitation_lambda"] = imaging_plane.excitation_lambda
            data["emission_lambda"] = optical_channel.emission_lambda

        # Add other metadata stored in nwb file to behavior ophys session meta
        data["experiment_datetime"] = self.nwbfile.session_start_time
        data["behavior_session_uuid"] = uuid.UUID(
            data["behavior_session_uuid"]
        )
        return data

    def get_cell_specimen_table(self) -> pd.DataFrame:
        # NOTE: ROI masks are stored in full frame width and height arrays
        df = (
            self.nwbfile.processing["ophys"]
            .data_interfaces["image_segmentation"]
            .plane_segmentations["cell_specimen_table"]
            .to_dataframe()
        )

        # Because pynwb stores this field as "image_mask", it is renamed here
        df = df.rename(columns={"image_mask": "roi_mask"})

        df.index.rename("cell_roi_id", inplace=True)
        df["cell_specimen_id"] = [
            None if csid == -1 else csid
            for csid in df["cell_specimen_id"].values
        ]

        df.reset_index(inplace=True)
        df.set_index("cell_specimen_id", inplace=True)

        if self.filter_invalid_rois:
            df = df[df["valid_roi"]]

        return df

    def get_dff_traces(self) -> pd.DataFrame:
        dff_nwb = (
            self.nwbfile.processing["ophys"]
            .data_interfaces["dff"]
            .roi_response_series["traces"]
        )
        # dff traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        dff_traces = dff_nwb.data[:].T
        number_of_cells, number_of_dff_frames = dff_traces.shape
        num_of_timestamps = len(self.get_ophys_timestamps())
        assert num_of_timestamps == number_of_dff_frames

        df = pd.DataFrame(
            {"dff": dff_traces.tolist()},
            index=pd.Index(data=dff_nwb.rois.table.id[:], name="cell_roi_id"),
        )
        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[["cell_roi_id"]].join(df, on="cell_roi_id")
        return df

    def get_corrected_fluorescence_traces(self) -> pd.DataFrame:
        corr_fluorescence_nwb = (
            self.nwbfile.processing["ophys"]
            .data_interfaces["corrected_fluorescence"]
            .roi_response_series["traces"]
        )
        # f traces stored as timepoints x rois in NWB
        # We want rois x timepoints, hence the transpose
        f_traces = corr_fluorescence_nwb.data[:].T
        df = pd.DataFrame(
            {"corrected_fluorescence": f_traces.tolist()},
            index=pd.Index(
                data=corr_fluorescence_nwb.rois.table.id[:], name="cell_roi_id"
            ),
        )

        cell_specimen_table = self.get_cell_specimen_table()
        df = cell_specimen_table[["cell_roi_id"]].join(df, on="cell_roi_id")
        return df

    def get_motion_correction(self) -> pd.DataFrame:
        ophys_module = self.nwbfile.processing["ophys"]

        motion_correction_data = {}
        motion_correction_data["x"] = ophys_module.get_data_interface(
            "ophys_motion_correction_x"
        ).data[:]
        motion_correction_data["y"] = ophys_module.get_data_interface(
            "ophys_motion_correction_y"
        ).data[:]

        return pd.DataFrame(motion_correction_data)

    def add_eye_tracking_data_to_nwb(
        self,
        nwbfile: NWBFile,
        eye_tracking_df: pd.DataFrame,
        eye_tracking_rig_geometry: Optional[dict],
        eye_gaze_mapping_file_path: Path = None,
    ) -> NWBFile:
        # 1. Add rig geometry
        if eye_tracking_rig_geometry:
            self.add_eye_tracking_rig_geometry_data_to_nwbfile(
                nwbfile=nwbfile,
                eye_tracking_rig_geometry=eye_tracking_rig_geometry,
            )

        # 2. Add eye gaze mapping
        if eye_gaze_mapping_file_path:
            eye_gaze_data = nwb.read_eye_gaze_mappings(
                Path(eye_gaze_mapping_file_path)
            )
            nwb.add_eye_gaze_mapping_data_to_nwbfile(
                nwbfile, eye_gaze_data=eye_gaze_data
            )

        # 3. Add eye tracking
        eye_tracking = EllipseSeries(
            name="eye_tracking",
            reference_frame="nose",
            data=eye_tracking_df[["eye_center_x", "eye_center_y"]].values,
            area=eye_tracking_df["eye_area"].values,
            width=eye_tracking_df["eye_width"].values,
            height=eye_tracking_df["eye_height"].values,
            angle=eye_tracking_df["eye_phi"].values,
            timestamps=eye_tracking_df["time"].values,
        )

        pupil_tracking = EllipseSeries(
            name="pupil_tracking",
            reference_frame="nose",
            data=eye_tracking_df[["pupil_center_x", "pupil_center_y"]].values,
            area=eye_tracking_df["pupil_area"].values,
            width=eye_tracking_df["pupil_width"].values,
            height=eye_tracking_df["pupil_height"].values,
            angle=eye_tracking_df["pupil_phi"].values,
            timestamps=eye_tracking,
        )

        corneal_reflection_tracking = EllipseSeries(
            name="corneal_reflection_tracking",
            reference_frame="nose",
            data=eye_tracking_df[["cr_center_x", "cr_center_y"]].values,
            area=eye_tracking_df["cr_area"].values,
            width=eye_tracking_df["cr_width"].values,
            height=eye_tracking_df["cr_height"].values,
            angle=eye_tracking_df["cr_phi"].values,
            timestamps=eye_tracking,
        )

        likely_blink = TimeSeries(
            timestamps=eye_tracking,
            data=eye_tracking_df["likely_blink"].values,
            name="likely_blink",
            description="blinks",
            unit="N/A",
        )

        ellipse_eye_tracking = EllipseEyeTracking(
            eye_tracking=eye_tracking,
            pupil_tracking=pupil_tracking,
            corneal_reflection_tracking=corneal_reflection_tracking,
            likely_blink=likely_blink,
        )

        nwbfile.add_acquisition(ellipse_eye_tracking)

        return nwbfile

    @staticmethod
    def add_eye_tracking_rig_geometry_data_to_nwbfile(
        nwbfile: NWBFile, eye_tracking_rig_geometry: dict
    ) -> NWBFile:
        """Rig geometry dict should consist of the following fields:
        monitor_position_mm: [x, y, z]
        monitor_rotation_deg: [x, y, z]
        camera_position_mm: [x, y, z]
        camera_rotation_deg: [x, y, z]
        led_position: [x, y, z]
        equipment: A string describing rig
        """
        eye_tracking_rig_mod = pynwb.ProcessingModule(
            name="eye_tracking_rig_metadata",
            description="Eye tracking rig metadata module",
        )

        ophys_eye_tracking_rig_metadata = load_pynwb_extension(
            OphysEyeTrackingRigMetadataSchema, "ndx-aibs-behavior-ophys"
        )

        rig_metadata = ophys_eye_tracking_rig_metadata(
            name="eye_tracking_rig_metadata",
            equipment=eye_tracking_rig_geometry["equipment"],
            monitor_position=eye_tracking_rig_geometry["monitor_position_mm"],
            monitor_position__unit_of_measurement="mm",
            camera_position=eye_tracking_rig_geometry["camera_position_mm"],
            camera_position__unit_of_measurement="mm",
            led_position=eye_tracking_rig_geometry["led_position"],
            led_position__unit_of_measurement="mm",
            monitor_rotation=eye_tracking_rig_geometry["monitor_rotation_deg"],
            monitor_rotation__unit_of_measurement="deg",
            camera_rotation=eye_tracking_rig_geometry["camera_rotation_deg"],
            camera_rotation__unit_of_measurement="deg",
        )

        eye_tracking_rig_mod.add_data_interface(rig_metadata)
        nwbfile.add_processing_module(eye_tracking_rig_mod)

        return nwbfile
