import logging
from datetime import datetime
import pytz
from typing import Optional

from allensdk.brain_observatory.behavior.session_apis.data_transforms import (
    BehaviorOphysDataXforms)


class BehaviorOphysJsonApi(BehaviorOphysDataXforms):
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

    def get_rig_name(self) -> str:
        """Get the name of the experiment rig (ex: CAM2P.3)"""
        return self.data['rig_name']

    def get_sex(self) -> str:
        """Get the sex of the subject (ex: 'M', 'F', or 'unknown')"""
        return self.data['sex']

    def get_age(self) -> str:
        """Get the age of the subject (ex: 'P15', 'Adult', etc...)"""
        return self.data['age']

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

    def get_stimulus_name(self) -> str:
        """Get the name of the stimulus presented for an ophys experiment"""
        return self.data['stimulus_name']

    def get_experiment_date(self) -> datetime:
        """Get the acquisition date of an ophys experiment"""
        return pytz.utc.localize(
            datetime.strptime(self.data['date_of_acquisition'],
                              "%Y-%m-%d %H:%M:%S"))

    def get_reporter_line(self) -> str:
        """Get the (gene) reporter line for the subject associated with an
        ophys experiment
        """
        return self.data['reporter_line']

    def get_driver_line(self) -> str:
        """Get the (gene) driver line for the subject associated with an ophys
        experiment"""
        return self.data['driver_line']

    def external_specimen_name(self) -> int:
        """Get the external specimen id (LabTracks ID) for the subject
        associated with an ophys experiment"""
        return self.data['external_specimen_name']

    def get_full_genotype(self) -> str:
        """Get the full genotype of the subject associated with an ophys
        experiment"""
        return self.data['full_genotype']

    def get_behavior_stimulus_file(self) -> str:
        """Get the filepath to the StimulusPickle file for the session"""
        return self.data['behavior_stimulus_file']

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

    def get_external_specimen_name(self) -> int:
        """Get the external specimen id for the subject associated with an
        ophys experiment"""
        return int(self.data['external_specimen_name'])

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
