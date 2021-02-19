import os
import logging
import sys
import argschema
import marshmallow

from allensdk.brain_observatory.behavior.behavior_ophys_session import (
    BehaviorOphysSession)
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi, BehaviorOphysJsonApi, BehaviorOphysLimsApi)
from allensdk.brain_observatory.behavior.write_nwb._schemas import (
    InputSchema, OutputSchema)
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs)
from allensdk.brain_observatory.session_api_utils import sessions_are_equal


def write_behavior_ophys_nwb(session_data, nwb_filepath):

    nwb_filepath_inprogress = nwb_filepath+'.inprogress'
    nwb_filepath_error = nwb_filepath+'.error'

    # Clean out files from previous runs:
    for filename in [nwb_filepath_inprogress,
                     nwb_filepath_error,
                     nwb_filepath]:
        if os.path.exists(filename):
            os.remove(filename)

    try:
        json_session = BehaviorOphysSession(
            api=BehaviorOphysJsonApi(session_data))
        lims_api = BehaviorOphysLimsApi(
            ophys_experiment_id=session_data['ophys_experiment_id'])
        lims_session = BehaviorOphysSession(api=lims_api)

        logging.info("Comparing a BehaviorOphysSession created from JSON "
                     "with a BehaviorOphysSession created from LIMS")
        assert sessions_are_equal(json_session, lims_session, reraise=True)

        BehaviorOphysNwbApi(nwb_filepath_inprogress).save(json_session)

        logging.info("Comparing a BehaviorOphysSession created from JSON "
                     "with a BehaviorOphysSession created from NWB")
        nwb_api = BehaviorOphysNwbApi(nwb_filepath_inprogress)
        nwb_session = BehaviorOphysSession(api=nwb_api)
        assert sessions_are_equal(json_session, nwb_session, reraise=True)

        os.rename(nwb_filepath_inprogress, nwb_filepath)
        return {'output_path': nwb_filepath}
    except Exception as e:
        os.rename(nwb_filepath_inprogress, nwb_filepath_error)
        raise e


def main():

    logging.basicConfig(
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')

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
        output = write_behavior_ophys_nwb(parser.args['session_data'],
                                          parser.args['output_path'])
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
