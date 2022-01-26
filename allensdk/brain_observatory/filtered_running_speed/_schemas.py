import argschema
from argschema.fields import Nested, String


class InputParameters(argschema.ArgSchema):
    output_path = argschema.fields.OutputFile(
        required=True,
        description="Filtered running speed hdf5 output file."
    )

    mapping_pkl_path = argschema.fields.InputFile(
        required=True,
        help="path to pkl file containing raw stimulus information",
    )
    behavior_pkl_path = argschema.fields.InputFile(
        required=True,
        allow_none=True,
        help="path to pkl file containing raw stimulus information",
    )
    replay_pkl_path = argschema.fields.InputFile(
        required=True,
        allow_none=True,
        help="path to pkl file containing raw stimulus information",
    )
    sync_h5_path = argschema.fields.InputFile(
        required=True,
        help="path to h5 file containing synchronization information",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OutputSchema(argschema.schemas.DefaultSchema):
    input_parameters = Nested(
        InputParameters,
        description=("Input parameters the module was run with"),
        required=True,
    )


class OutputParameters(OutputSchema):
    output_path = String(required=True, help="path to output file")
