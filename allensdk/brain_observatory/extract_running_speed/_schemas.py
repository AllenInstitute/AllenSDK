from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int


class InputParameters(ArgSchema):
    output_running_speed_path = String(required=True, help="write outputs to here")
    stimulus_pkl_path = String(
        required=True,
        help="path to pkl file containing raw stimulus information",
    )
    sync_h5_path = String(
        required=True,
        help="path to h5 file containing synchronization information",
    )
    wheel_radius = Float(default=8.255, help="radius, in cm, of running wheel")
    subject_position = Float(
        default=2 / 3,
        help="normalized distance of the subject from the center of the running wheel (1 is rim, 0 is center)",
    )


class OutputSchema(DefaultSchema):
    input_parameters = Nested(
        InputParameters,
        description=("Input parameters the module " "was run with"),
        required=True,
    )


class OutputParameters(OutputSchema):
    output_path = String(required=True, help="path to output file")
