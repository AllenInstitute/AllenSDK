import hashlib

from marshmallow import RAISE, ValidationError

from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import LogLevel, String, Int, DateTime, Nested, Boolean, Float

from ..argschema_utilities import check_read_access, check_write_access, RaisingSchema

available_hashers = {
    'sha3_256': hashlib.sha3_256,
    None: None
}

class FileToCopy(RaisingSchema):
    source = String(required=True, validate=check_read_access, description='copy from here')
    destination = String(required=True, validation=check_write_access, description='copy to here (full path, not just directory!)')
    key = String(required=False, description='will be passed through to outputs, allowing a name or kind to be associated with this file')


class CopiedFile(RaisingSchema):
    source = String(required=True, description='copied from here')
    destination = String(required=True, description='copied to here')
    key = String(required=False, description='passed from inputs')
    source_hash = String(required=False) # string vs bytes for JSONability
    destination_hash = String(required=False)


class InputSchema(ArgSchema):
    class Meta:
        unknown=RAISE
    log_level = LogLevel(default='INFO', description='set the logging level of the module')
    files = Nested(FileToCopy, required=True, description='files to be copied')
    use_rsync = Boolean(default=True, 
        description='copy files using rsync rather than shutil (this is not likely to work if you are running windows!)'
    )
    hasher_key = String(default=None, validate=lambda st: st in available_hashers, 
        description='select a hash function to compute over base64-encoded pre- and post-copy files'
    )
    raise_if_comparison_fails = Boolean(default=True, description='if a hash comparison fails, throw an error (vs. a warning)')
    make_parent_dirs = Boolean(default=True, description='build missing parent directories for destination')
    

class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')
    