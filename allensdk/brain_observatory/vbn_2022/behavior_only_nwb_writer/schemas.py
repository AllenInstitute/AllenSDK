import argschema
import pathlib
from marshmallow import post_load


class VBN2022BehaviorOnlyWriterSchema(argschema.ArgSchema):

    behavior_session_id_list = argschema.fields.List(
            argschema.fields.Int,
            required=True,
            description=("List of behavior_session_id values "
                         "of behavior only sessions to be released"))

    behavior_session_table = argschema.fields.InputFile(
            required=True,
            description=("CSV file containing information on the behavior "
                         "only sessions to be released."))

    nwb_output_dir = argschema.fields.OutputDir(
            required=True,
            description=("Directory to write the Behavior only NWBs to."))

    lims_user = argschema.fields.String(
            required=True,
            description=("Username for LIMS2"))

    lims_password = argschema.fields.String(
            required=True,
            description=("Password for LIMS2 login."))
