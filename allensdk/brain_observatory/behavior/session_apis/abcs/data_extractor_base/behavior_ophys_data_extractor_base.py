import abc
from typing import Dict, Optional

from allensdk.brain_observatory.behavior.session_apis.abcs.\
    data_extractor_base.behavior_data_extractor_base import \
    BehaviorDataExtractorBase


class BehaviorOphysDataExtractorBase(BehaviorDataExtractorBase):
    """Abstract base class implementing required methods for extracting
    data (from LIMS or from JSON) that will be transformed or passed on to
    fill behavior + ophys session data.
    """

    @abc.abstractmethod
    def get_ophys_experiment_id(self) -> int:
        """Return the ophys experiment id (experiments are an internal alias
        for an imaging plane)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ophys_session_id(self) -> int:
        """Return the ophys session id"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_surface_2p_pixel_size_um(self) -> float:
        """Get the pixel size for 2-photon movies in micrometers"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_sync_file(self) -> str:
        """Get the filepath of the sync timing file associated with the
        ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_field_of_view_shape(self) -> Dict[str, int]:
        """Get a field of view dictionary for a given ophys experiment.
           ex: {"width": int, "height": int}
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ophys_container_id(self) -> int:
        """Get the experiment container id associated with an ophys
        experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_targeted_structure(self) -> str:
        """Get the targeted structure (acronym) for an ophys experiment
        (ex: "Visp")"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_imaging_depth(self) -> int:
        """Get the imaging depth for an ophys experiment
        (ex: 400, 500, etc.)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dff_file(self) -> str:
        """Get the filepath of the dff trace file associated with an ophys
        experiment."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_event_detection_filepath(self) -> str:
        """Get the filepath of the .h5 events file associated with an ophys
        experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ophys_cell_segmentation_run_id(self) -> int:
        """Get the ophys cell segmentation run id associated with an
        ophys experiment id"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_raw_cell_specimen_table_dict(self) -> dict:
        """Get the cell_rois table from LIMS in dictionary form"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_demix_file(self) -> str:
        """Get the filepath of the demixed traces file associated with an
        ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_average_intensity_projection_image_file(self) -> str:
        """Get the avg intensity project image filepath associated with an
        ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_max_projection_file(self) -> str:
        """Get the filepath of the max projection image associated with the
        ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rigid_motion_transform_file(self) -> str:
        """Get the filepath for the motion transform file (.csv) associated
        with an ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_imaging_plane_group(self) -> Optional[int]:
        """Get the imaging plane group number. This is a numeric index
        that indicates the order that the frames were acquired when
        there is more than one frame acquired concurrently. Relevant for
        mesoscope data timestamps, as the laser jumps between plane
        groups during the scan. Will be None for non-mesoscope data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_plane_group_count(self) -> int:
        """Gets the total number of plane groups in the session.
        This is required for resampling ophys timestamps for mesoscope
        data. Will be 0 if the scope did not capture multiple concurrent
        frames. See `get_imaging_plane_group` for more info.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_eye_tracking_rig_geometry(self) -> dict:
        """Get the eye tracking rig geometry associated with an
        ophys experiment"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_eye_tracking_filepath(self) -> dict:
        """Get the eye tracking filepath containing ellipse fits"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_project_code(self) -> str:
        """Get the project code."""
        raise NotImplementedError()
