import datetime
import pytz
import os
import logging
import sys
import argparse
import json
import argschema
import marshmallow

from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi, equals
from allensdk.brain_observatory.behavior.write_nwb._schemas import InputSchema, OutputSchema
from allensdk.brain_observatory.argschema_utilities import write_or_print_outputs


class BehaviorOphysJsonApi(BehaviorOphysLimsApi):

    def __init__(self, data):
        self.data = data

    def get_ophys_experiment_id(self):
        return self.data['ophys_experiment_id']

    def get_surface_2p_pixel_size_um(self):
        return self.data['surface_2p_pixel_size_um']

    def get_max_projection_file(self):
        return self.data['max_projection_file']

    def get_segmentation_mask_image_file(self):
        return self.data['segmentation_mask_image_file']

    def get_sync_file(self):
        return self.data['sync_file']

    def get_rig_name(self):
        return self.data['rig_name']

    def get_sex(self):
        return self.data['sex']

    def get_age(self):
        return self.data['age']

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


def write_behavior_ophys_nwb(session_data, nwb_filepath):

    nwb_filepath_inprogress = nwb_filepath+'.inprogress'
    nwb_filepath_error = nwb_filepath+'.error'

    # Clean out files from previous runs:
    for filename in [nwb_filepath_inprogress, nwb_filepath_error, nwb_filepath]:
        if os.path.exists(filename):
            os.remove(filename)

    try:
        session = BehaviorOphysSession(api=BehaviorOphysJsonApi(session_data))
        BehaviorOphysNwbApi(nwb_filepath_inprogress).save(session)
        assert equals(session, BehaviorOphysSession(api=BehaviorOphysNwbApi(nwb_filepath_inprogress)))
        os.rename(nwb_filepath_inprogress, nwb_filepath)
        return {'output_path': nwb_filepath}
    except Exception as e:
        os.rename(nwb_filepath_inprogress, nwb_filepath_error)
        raise e



def main():

    logging.basicConfig(format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')

    args = sys.argv[1:]
    try:
        parser = argschema.ArgSchemaParser(
            args=args,
            schema_type=InputSchema,
            output_schema_type=OutputSchema,
        )
        logging.info('Input successfully parsed')
    except marshmallow.exceptions.ValidationError as err:
        logging.error('Parsing failure')
        print(err)
        raise err

    try:
        output = write_behavior_ophys_nwb(parser.args['session_data'], parser.args['output_path'])
        logging.info('File successfully created')
    except Exception as err:
        logging.error('NWB write failure')
        print(err)
        raise err

    write_or_print_outputs(output, parser)


if __name__ == "__main__":

    # input_dict = {'log_level':'DEBUG',
    #               'session_data': {'ophys_experiment_id': 789359614,
    #                                 'surface_2p_pixel_size_um': 0.78125,
    #                                 "max_projection_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/maxInt_a13a.png",
    #                                 "sync_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/789220000_sync.h5",
    #                                 "rig_name": "CAM2P.5",
    #                                 "movie_width": 447,
    #                                 "movie_height": 512,
    #                                 "container_id": 814796558,
    #                                 "targeted_structure": "VISp",
    #                                 "targeted_depth": 375,
    #                                 "stimulus_name": "Unknown",
    #                                 "date_of_acquisition": '2018-11-30 23:28:37',
    #                                 "reporter_line": ["Ai93(TITL-GCaMP6f)"],
    #                                 "driver_line": ['Camk2a-tTA', 'Slc17a7-IRES2-Cre'],
    #                                 "external_specimen_name": 416369,
    #                                 "full_genotype": "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt",
    #                                 "behavior_stimulus_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl",
    #                                 "dff_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/789359614_dff.h5",
    #                                 "ophys_cell_segmentation_run_id": 789410052,
    #                                 "cell_specimen_table_dict": json.load(open('/home/nicholasc/projects/allensdk/allensdk/test/brain_observatory/behavior/cell_specimen_table_789359614.json', 'r')),
    #                                 "demix_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/demix/789359614_demixed_traces.h5",
    #                                 "average_intensity_projection_image_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/ophys_cell_segmentation_run_789410052/avgInt_a1X.png",
    #                                 "rigid_motion_transform_file": "/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/ophys_experiment_789359614/processed/789359614_rigid_motion_transform.csv",
    #                                 },
    #                 'output_path': 'tmp.nwb'}
    # json.dump(input_dict, open('dev.json', 'w'))

    main()
