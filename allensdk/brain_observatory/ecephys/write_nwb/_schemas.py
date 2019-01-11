from marshmallow import RAISE
from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema


class RaisingSchema(DefaultSchema):
    class META:
        unknown=RAISE


class InputParameters(ArgSchema, RaisingSchema):
    log_level = LogLevel(default='INFO',description="set the logging level of the module")


class OutputSchema(RaisingSchema):
    input_parameters = Nested(InputParameters, 
                              description=("Input parameters the module " 
                                           "was run with"), 
                              required=True) 