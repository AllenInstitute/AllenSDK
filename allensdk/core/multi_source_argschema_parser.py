import argparse
import json

from argschema import ArgSchemaParser


class MultiSourceArgschemaParser(ArgSchemaParser):

    def __init__(self, sources, print_inputs=True, **pass_kwargs):

        parser = argparse.ArgumentParser()
        parser.add_argument('--source', default=None)
        for source, info in sources.items():
            for param, default in info['params'].items():
                if not param[:2] == '--':
                    param = '--' + param
                parser.add_argument(param, default=default)

        source_args, other_args = parser.parse_known_args()

        if source_args.source is not None:
            source = sources[source_args.source]
            input_data_args = dict(source_args.__dict__)
            del input_data_args['source']
            input_data = source['get_input_data'](**input_data_args)

            super(MultiSourceArgschemaParser, self).__init__(
                input_data=input_data,
                args=other_args,
                **pass_kwargs
            )

        else:
            super(MultiSourceArgschemaParser, self).__init__(
                args=other_args,
                **pass_kwargs
            )

        if print_inputs:
            print(json.dumps(self.args, indent=2))


    @classmethod
    def write_or_print_outputs(cls, data, parser):
        data.update({'input_parameters': parser.args})
        if 'output_json' in parser.args:
            parser.output(data, indent=2)
        else:
            print(parser.get_output_json(data))