import logging
from typing import Optional

from allensdk.brain_observatory.behavior.session_apis.abcs.\
    data_extractor_base.behavior_ophys_data_extractor_base import \
    BehaviorOphysDataExtractorBase
from allensdk.brain_observatory.behavior.session_apis.data_io import \
    BehaviorJsonExtractor
from allensdk.brain_observatory.behavior.session_apis.data_transforms import \
    BehaviorOphysDataTransforms


class BehaviorOphysJsonApi(BehaviorOphysDataTransforms):
    """A data fetching and processing class that serves processed data from
    a specified raw data source (extractor). Contains all methods
    needed to fill a BehaviorOphysExperiment."""

    def __init__(self, data: dict, skip_eye_tracking: bool = False):
        extractor = BehaviorOphysJsonExtractor(data=data)
        super().__init__(extractor=extractor,
                         skip_eye_tracking=skip_eye_tracking)


class BehaviorOphysJsonExtractor(BehaviorJsonExtractor,
                                 BehaviorOphysDataExtractorBase):
    """A class which 'extracts' data from a json file. The extracted data
    is necessary (but not sufficient) for populating a 'BehaviorOphysExperiment'.

    Most data provided by this extractor needs to be processed by
    BehaviorOphysDataTransforms methods in order to usable by
    'BehaviorOphysExperiment's.

    This class is used by the write_nwb module for behavior ophys sessions.
    """

    def __init__(self, data):
        super().__init__(data)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_ophys_experiment_id(self) -> int:
        return self.data['ophys_experiment_id']

    def get_ophys_session_id(self):
        return self.data['ophys_session_id']

    def get_surface_2p_pixel_size_um(self) -> float:
        """Get the pixel size for 2-photon movies in micrometers"""
        return self.data['surface_2p_pixel_size_um']

    def get_max_projection_file(self) -> str:
        """Get the filepath of the max projection image associated with the
        ophys experiment"""
        return self.data['max_projection_file']

    def get_sync_file(self) -> str:
        """Get the filepath of the sync timing file associated with the
        ophys experiment"""
        return self.data['sync_file']

    def get_field_of_view_shape(self) -> dict:
        """Get a field of view dictionary for a given ophys experiment.
           ex: {"width": int, "height": int}
        """
        return {'height': self.data['movie_height'],
                'width': self.data['movie_width']}

    def get_ophys_container_id(self) -> int:
        """Get the experiment container id associated with an ophys
        experiment"""
        return self.data['container_id']

    def get_targeted_structure(self) -> str:
        """Get the targeted structure (acronym) for an ophys experiment
        (ex: "Visp")"""
        return self.data['targeted_structure']

    def get_imaging_depth(self) -> int:
        """Get the imaging depth for an ophys experiment
        (ex: 400, 500, etc.)"""
        return self.data['targeted_depth']

    def get_dff_file(self) -> str:
        """Get the filepath of the dff trace file associated with an ophys
        experiment."""
        return self.data['dff_file']

    def get_ophys_cell_segmentation_run_id(self) -> int:
        """Get the ophys cell segmentation run id associated with an
        ophys experiment id"""
        return self.data['ophys_cell_segmentation_run_id']

    def get_raw_cell_specimen_table_dict(self) -> dict:
        """Get the cell_rois table from LIMS in dictionary form"""
        return self.data['cell_specimen_table_dict']

    def get_demix_file(self) -> str:
        """Get the filepath of the demixed traces file associated with an
        ophys experiment"""
        return self.data['demix_file']

    def get_average_intensity_projection_image_file(self) -> str:
        """Get the avg intensity project image filepath associated with an
        ophys experiment"""
        return self.data['average_intensity_projection_image_file']

    def get_rigid_motion_transform_file(self) -> str:
        """Get the filepath for the motion transform file (.csv) associated
        with an ophys experiment"""
        return self.data['rigid_motion_transform_file']

    def get_imaging_plane_group(self) -> Optional[int]:
        """Get the imaging plane group number. This is a numeric index
        that indicates the order that the frames were acquired when
        there is more than one frame acquired concurrently. Relevant for
        mesoscope data timestamps, as the laser jumps between plane
        groups during the scan. Will be None for non-mesoscope data.
        """
        return self.data["imaging_plane_group"]

    def get_plane_group_count(self) -> int:
        """Gets the total number of plane groups in the session.
        This is required for resampling ophys timestamps for mesoscope
        data. Will be 0 if the scope did not capture multiple concurrent
        frames (e.g. data from Scientifica microscope).
        """
        return self.data["plane_group_count"]

    def get_eye_tracking_rig_geometry(self) -> dict:
        """Get the eye tracking rig geometry associated with an ophys
        experiment"""
        return self.data['eye_tracking_rig_geometry']

    def get_eye_tracking_filepath(self) -> dict:
        """Get the eye tracking filepath containing ellipse fits"""
        return self.data['eye_tracking_filepath']

    def get_event_detection_filepath(self) -> str:
        """Get the filepath of the .h5 events file associated with an ophys
        experiment"""
        return self.data['events_file']

    def get_project_code(self) -> str:
        raise NotImplementedError('Not exposed externally')
