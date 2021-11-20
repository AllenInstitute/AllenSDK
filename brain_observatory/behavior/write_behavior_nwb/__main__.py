import os
import logging
import sys
import argschema
import marshmallow
from pynwb import NWBHDF5IO

from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.write_behavior_nwb._schemas import (
    InputSchema, OutputSchema)
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs)
from allensdk.brain_observatory.session_api_utils import sessions_are_equal


def write_behavior_nwb(session_data, nwb_filepath):

    nwb_filepath_inprogress = nwb_filepath+'.inprogress'
    nwb_filepath_error = nwb_filepath+'.error'

    # Clean out files from previous runs:
    for filename in [nwb_filepath_inprogress,
                     nwb_filepath_error,
                     nwb_filepath]:
        if os.path.exists(filename):
            os.remove(filename)

    try:
        json_session = BehaviorSession.from_json(session_data)

        behavior_session_id = session_data['behavior_session_id']
        lims_session = BehaviorSession.from_lims(behavior_session_id)

        logging.info("Comparing a BehaviorSession created from JSON "
                     "with a BehaviorSession created from LIMS")
        assert sessions_are_equal(json_session, lims_session, reraise=True)

        nwbfile = lims_session.to_nwb()
        with NWBHDF5IO(nwb_filepath_inprogress, 'w') as nwb_file_writer:
            nwb_file_writer.write(nwbfile)

        logging.info("Comparing a BehaviorSession created from JSON "
                     "with a BehaviorSession created from NWB")
        nwb_session = BehaviorSession.from_nwb_path(nwb_filepath_inprogress)
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
        output = write_behavior_nwb(parser.args['session_data'],
                                    parser.args['output_path'])
        logging.info('File successfully created')
    except Exception as err:
        logging.error('NWB write failure')
        print(err)
        raise err

    write_or_print_outputs(output, parser)


if __name__ == "__main__":
    main()
