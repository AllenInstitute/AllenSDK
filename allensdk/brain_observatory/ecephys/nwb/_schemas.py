from marshmallow import RAISE
from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.Fields import LogLevel, String, Int, DateTime


class RaisingSchema(DefaultSchema):
    class Meta:
        unknown=RAISE


class InputSchema(ArgSchema):
    class Meta:
        unknown=RAISE
    log_level = LogLevel(default='INFO', description='set the logging level of the module')
    session_id = Int(required=True, description='unique identifier for this ecephys session')
    session_start_time = DateTime(required=True, description='the date and time (UTC, timezone aware) at which the session started')


class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')