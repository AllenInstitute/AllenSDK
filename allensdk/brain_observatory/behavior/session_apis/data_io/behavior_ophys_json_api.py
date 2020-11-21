import logging
from datetime import datetime
import pytz

from allensdk.brain_observatory.behavior.session_apis.data_transforms import (
    BehaviorOphysDataXforms)


class BehaviorOphysJsonApi(BehaviorOphysDataXforms):
    """
        This class is used by both Scientifica and Mesoscope ophys experiments.
    """

    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_ophys_experiment_id(self):
        return self.data['ophys_experiment_id']

    # TODO: This should be replaced with a dict lookup after the
    # behavior_ophys_write_nwb LIMS strategy has been updated
    def get_behavior_session_id(self):
        NotImplementedError()

    # TODO: This should be replaced with a dict lookup after the
    # behavior_ophys_write_nwb LIMS strategy has been updated
    def get_ophys_session_id(self):
        NotImplementedError()

    def get_surface_2p_pixel_size_um(self):
        return self.data['surface_2p_pixel_size_um']

    def get_max_projection_file(self):
        return self.data['max_projection_file']

    def get_sync_file(self):
        return self.data['sync_file']

    def get_rig_name(self):
        return self.data['rig_name']

    def get_sex(self):
        return self.data['sex']

    def get_age(self):
        return self.data['age']

    def get_field_of_view_shape(self):
        return {'height': self.data['movie_height'],
                'width': self.data['movie_width']}

    def get_experiment_container_id(self):
        return self.data['container_id']

    def get_targeted_structure(self):
        return self.data['targeted_structure']

    def get_imaging_depth(self):
        return self.data['targeted_depth']

    def get_stimulus_name(self):
        return self.data['stimulus_name']

    def get_experiment_date(self):
        return pytz.utc.localize(
            datetime.strptime(self.data['date_of_acquisition'],
                              "%Y-%m-%d %H:%M:%S"))

    def get_reporter_line(self):
        return self.data['reporter_line']

    def get_driver_line(self):
        return self.data['driver_line']

    def external_specimen_name(self):
        return self.data['external_specimen_name']

    def get_full_genotype(self):
        return self.data['full_genotype']

    def get_behavior_stimulus_file(self):
        return self.data['behavior_stimulus_file']

    def get_dff_file(self):
        return self.data['dff_file']

    def get_ophys_cell_segmentation_run_id(self):
        return self.data['ophys_cell_segmentation_run_id']

    def get_raw_cell_specimen_table_dict(self):
        return self.data['cell_specimen_table_dict']

    def get_demix_file(self):
        return self.data['demix_file']

    def get_average_intensity_projection_image_file(self):
        return self.data['average_intensity_projection_image_file']

    def get_rigid_motion_transform_file(self):
        return self.data['rigid_motion_transform_file']

    def get_external_specimen_name(self):
        return self.data['external_specimen_name']

    def get_imaging_plane_group(self):
        try:
            # Will only contain the "imaging_plane_group" key if we are
            # dealing with Mesoscope data
            return self.data["imaging_plane_group"]
        except KeyError:
            return None
