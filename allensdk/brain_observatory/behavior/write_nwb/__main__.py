import os
import logging
import sys
import argschema
import marshmallow
from pynwb import NWBHDF5IO

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)
from allensdk.brain_observatory.behavior.write_nwb._schemas import (
    InputSchema, OutputSchema)
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs)
from allensdk.brain_observatory.session_api_utils import sessions_are_equal


def write_behavior_ophys_nwb(session_data: dict,
                             nwb_filepath: str,
                             skip_eye_tracking: bool):

    nwb_filepath_inprogress = nwb_filepath+'.inprogress'
    nwb_filepath_error = nwb_filepath+'.error'

    # Clean out files from previous runs:
    for filename in [nwb_filepath_inprogress,
                     nwb_filepath_error,
                     nwb_filepath]:
        if os.path.exists(filename):
            os.remove(filename)

    try:
        json_session = BehaviorOphysExperiment.from_json(
            session_data=session_data, skip_eye_tracking=skip_eye_tracking)
        lims_session = BehaviorOphysExperiment.from_lims(
            ophys_experiment_id=session_data['ophys_experiment_id'],
            skip_eye_tracking=skip_eye_tracking)

        logging.info("Comparing a BehaviorOphysExperiment created from JSON "
                     "with a BehaviorOphysExperiment created from LIMS")
        assert sessions_are_equal(json_session, lims_session, reraise=True,
                                  ignore_keys={'metadata': {'project_code'}})

        nwbfile = json_session.to_nwb()
        with NWBHDF5IO(nwb_filepath_inprogress, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        logging.info("Comparing a BehaviorOphysExperiment created from JSON "
                     "with a BehaviorOphysExperiment created from NWB")
        nwb_session = BehaviorOphysExperiment.from_nwb(nwbfile=nwbfile)
        assert sessions_are_equal(json_session, nwb_session, reraise=True)

        os.rename(nwb_filepath_inprogress, nwb_filepath)
        return {'output_path': nwb_filepath}
    except Exception as e:
        if os.path.isfile(nwb_filepath_inprogress):
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
        skip_eye_tracking = parser.args['skip_eye_tracking']
        output = write_behavior_ophys_nwb(parser.args['session_data'],
                                          parser.args['output_path'],
                                          skip_eye_tracking)
        logging.info('File successfully created')
    except Exception as err:
        logging.error('NWB write failure')
        print(err)
        raise err

    write_or_print_outputs(output, parser)


if __name__ == "__main__":
    main()
