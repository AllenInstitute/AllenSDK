from allensdk.brain_observatory.behavior.session_apis.abcs import (
    BehaviorRawDataBase)

from typing import Dict, Optional


class BehaviorOphysRawDataBase(BehaviorRawDataBase):
    """Abstract base class implementing required methods for interacting with
    raw (from LIMS or from JSON) behavior + ophys session data.
    """

    def get_ophys_experiment_id(self) -> int:
        """Return the ophys experiment id (experiments are an internal alias
        for an imaging plane)"""
        raise NotImplementedError()

    def get_ophys_session_id(self) -> int:
        """Return the ophys session id"""
        raise NotImplementedError()

    def get_surface_2p_pixel_size_um(self) -> float:
        """Get the pixel size for 2-photon movies in micrometers"""
        raise NotImplementedError()

    def get_sync_file(self) -> str:
        """Get the filepath of the sync timing file associated with the
        ophys experiment"""
        raise NotImplementedError()

    def get_field_of_view_shape(self) -> Dict[str, int]:
        """Get a field of view dictionary for a given ophys experiment.
           ex: {"width": int, "height": int}
        """
        raise NotImplementedError()

    def get_experiment_container_id(self) -> int:
        """Get the experiment container id associated with an ophys
        experiment"""
        raise NotImplementedError()

    def get_targeted_structure(self) -> str:
        """Get the targeted structure (acronym) for an ophys experiment
        (ex: "Visp")"""
        raise NotImplementedError()

    def get_imaging_depth(self) -> int:
        """Get the imaging depth for an ophys experiment
        (ex: 400, 500, etc.)"""
        raise NotImplementedError()

    def get_dff_file(self) -> str:
        """Get the filepath of the dff trace file associated with an ophys
        experiment."""
        raise NotImplementedError()

    def get_ophys_cell_segmentation_run_id(self) -> int:
        """Get the ophys cell segmentation run id associated with an
        ophys experiment id"""
        raise NotImplementedError()

    def get_raw_cell_specimen_table_dict(self) -> dict:
        """Get the cell_rois table from LIMS in dictionary form"""
        raise NotImplementedError()

    def get_demix_file(self) -> str:
        """Get the filepath of the demixed traces file associated with an
        ophys experiment"""
        raise NotImplementedError()

    def get_average_intensity_projection_image_file(self) -> str:
        """Get the avg intensity project image filepath associated with an
        ophys experiment"""
        raise NotImplementedError()

    def get_max_projection_file(self) -> str:
        """Get the filepath of the max projection image associated with the
        ophys experiment"""
        raise NotImplementedError()

    def get_rigid_motion_transform_file(self) -> str:
        """Get the filepath for the motion transform file (.csv) associated
        with an ophys experiment"""
        raise NotImplementedError()

    def get_imaging_plane_group(self) -> Optional[int]:
        """Get the imaging plane group number. This is a numeric index
        that indicates the order that the frames were acquired when
        there is more than one frame acquired concurrently. Relevant for
        mesoscope data timestamps, as the laser jumps between plane
        groups during the scan. Will be None for non-mesoscope data.
        """
        raise NotImplementedError()

    def get_plane_group_count(self) -> int:
        """Gets the total number of plane groups in the session.
        This is required for resampling ophys timestamps for mesoscope
        data. Will be 0 if the scope did not capture multiple concurrent
        frames. See `get_imaging_plane_group` for more info.
        """
        raise NotImplementedError()

    def get_eye_tracking_rig_geometry(self) -> dict:
        """Get the eye tracking rig geometry associated with an
        ophys experiment"""
        raise NotImplementedError()

    def get_eye_tracking_filepath(self) -> dict:
        """Get the eye tracking filepath containing ellipse fits"""
        raise NotImplementedError()

    def get_eye_gaze_mapping_file_path(self) -> str:
        """Get h5 filepath containing eye gaze behavior of the experiment's
        subject"""
        raise NotImplementedError()
