import logging
import sys
import argschema
import marshmallow
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession
from allensdk.brain_observatory.nwb.nwb_utils import NWBWriter
from allensdk.brain_observatory.ecephys.write_nwb.vbn._schemas import \
    VBNInputSchema, OutputSchema


def main():
    args = sys.argv[1:]
    try:
        parser = argschema.ArgSchemaParser(
            args=args,
            schema_type=VBNInputSchema,
            output_schema_type=OutputSchema,
        )
        logging.info('Input successfully parsed')
    except marshmallow.exceptions.ValidationError as err:
        logging.error('Parsing failure')
        logging.error(err)
        raise err

    nwb_writer = NWBWriter(
        nwb_filepath=parser.args['output_path'],
        session_data=parser.args['session_data'],
        serializer=BehaviorEcephysSession
    )

    try:
        nwb_writer.write_nwb()
        logging.info('File successfully created')
    except Exception as err:
        logging.error('NWB write failure')
        logging.error(err)
        raise err


if __name__ == "__main__":
    main()
