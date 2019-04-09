import datetime
import pytz
import json
import os

from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession


class BehaviorOphysJsonApi(BehaviorOphysLimsApi):

    def __init__(self, data):
        self.data = data

    def get_ophys_experiment_id(self):
        return self.data['ophys_experiment_id']

    def get_surface_2p_pixel_size_um(self):
        return self.data['surface_2p_pixel_size_um']

    def get_segmentation_mask_image_file(self):
        return self.data['segmentation_mask_image_file']

    def get_sync_file(self):
        return self.data['sync_file']

    def get_rig_name(self):
        return self.data['rig_name']

    def get_field_of_view_shape(self):
        return {'height': self.data['movie_height'], 'width': self.data['movie_width']}

    def get_experiment_container_id(self):
        return self.data['container_id']

    def get_targeted_structure(self):
        return self.data['targeted_structure']

    def get_imaging_depth(self):
        return self.data['targeted_depth']

    def get_stimulus_name(self):
        return self.data['stimulus_name']

    def get_experiment_date(self):
        return pytz.utc.localize(datetime.datetime.strptime(self.data['date_of_acquisition'], "%Y-%m-%d %H:%M:%S"))

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

    def get_raw_cell_specimen_table_json(self):
        return self.data['cell_specimen_table']

    def get_demix_file(self):
        return self.data['demix_file']

    def get_average_intensity_projection_image(self):
        return self.data['average_intensity_projection_image']

    def get_rigid_motion_transform_file(self):
        return self.data['rigid_motion_transform_file']

    def get_external_specimen_name(self):
        return self.data['external_specimen_name']


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
