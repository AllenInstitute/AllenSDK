from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int


class InputParameters(ArgSchema):
    opto_pickle_path = String(required=True, help='path to file containing optotagging information')
    sync_h5_path = String(required=True, help='path to h5 file containing syncronization information')
    output_opto_table_path = String(required=True, help='the optotagging stimulation table will be written here')    


class OutputSchema(DefaultSchema):
    input_parameters = Nested(InputParameters, description=('Input parameters the module was run with'), required=True) 


class OutputParameters(OutputSchema):
    output_opto_table_path = String(required=True, help='path to optotagging stimulation table')