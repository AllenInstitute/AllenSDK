
from argschema import ArgSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int, List

from allensdk.core.multi_source_argschema_parser import MultiSourceArgschemaParser


class InputParameters(ArgSchema):
    an_int = Int()
    a_float = Float()
    a_string = String()
    not_in_lims = Int()


def get_inputs_from_lims(an_int, a_float, a_string):
    return {
        'an_int': an_int,
        'a_float': a_float,
        'a_string': a_string,
        'not_in_lims': 9
    }


def main():

    parser = MultiSourceArgschemaParser(
        {
            'lims': {
                'get_input_data': get_inputs_from_lims,
                'params': {
                    'an_int': None,
                    'a_float': None,
                    'a_string': 'hello world'
                }
            }
        },
        schema_type=InputParameters
    )



if __name__ == '__main__':
    main()