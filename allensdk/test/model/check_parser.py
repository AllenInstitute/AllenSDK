from allensdk.model.biophysical.runner import sim_parser

def get_parsed_args(schema):
    print(vars(schema))

if __name__ == '__main__':
    schema = sim_parser.parse_args()
    get_parsed_args(schema)
