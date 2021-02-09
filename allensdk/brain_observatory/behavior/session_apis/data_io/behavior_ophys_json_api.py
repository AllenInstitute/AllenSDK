import logging
from typing import Optional

from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorJsonApi)
from allensdk.brain_observatory.behavior.session_apis.data_transforms import (
    BehaviorOphysDataXforms)


class BehaviorOphysJsonApi(BehaviorOphysDataXforms, BehaviorJsonApi):
    """A data fetching class that serves as an API for fetching 'raw'
    data from a json file necessary (but not sufficient) for filling
    a 'BehaviorOphysSession'.

    Most 'raw' data provided by this API needs to be processed by
    BehaviorOphysDataXforms methods in order to usable by
    'BehaviorOphysSession's.

    This class is used by the write_nwb module for behavior ophys sessions.
    """

    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_ophys_experiment_id(self) -> int:
        return self.data['ophys_experiment_id']

    # TODO: This should be replaced with a dict lookup after the
    # behavior_ophys_write_nwb LIMS strategy has been updated
    def get_behavior_session_id(self):
        NotImplementedError()

    # TODO: This should be replaced with a dict lookup after the
    # behavior_ophys_write_nwb LIMS strategy has been updated
    def get_ophys_session_id(self):
        NotImplementedError()

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

    def get_experiment_container_id(self) -> int:
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
        try:
            # Will only contain the "imaging_plane_group" key if we are
            # dealing with Mesoscope data
            return self.data["imaging_plane_group"]
        except KeyError:
            return None

    def get_eye_tracking_rig_geometry(self) -> dict:
        """Get the eye tracking rig geometry associated with an ophys experiment"""
        return self.data['eye_tracking_rig_geometry']

    def get_eye_tracking_filepath(self) -> dict:
        """Get the eye tracking filepath containing ellipse fits"""
        return self.data['eye_tracking_filepath']

    def get_eye_gaze_mapping_file_path(self) -> str:
        """Get h5 filepath containing eye gaze behavior of the experiment's subject"""
        return self.data['eye_gaze_mapping_path']