from dataclasses import dataclass

import pandas as pd

from allensdk.brain_observatory.argschema_utilities import ArgSchemaParserPlus
from allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset import (
    EcephysSyncDataset,
)
from ._schemas import InputParameters, OutputParameters


@dataclass
class OptoCondition:
    duration: float
    name: str
    description: str


kwown_conditions = {
    0: OptoCondition(duration=1.0, name="fast_pulses", description="2.5 ms pulses at 10 Hz"),
    1: OptoCondition(duration=0.005, name="single_pulse", description="a single square pulse"),
    2: OptoCondition(duration=0.01, name="single_pulse", description="a single square pulse"),
    3: OptoCondition(duration=1.0, name="raised_cosine", description="a cosine wave"),
}


def build_opto_table(args):

    opto_file = pd.read_pickle(args['opto_pickle_path'])
    sync_file = EcephysSyncDataset.factory(args['sync_h5_path'])

    optotagging_table = pd.DataFrame({
        'start_time': sync_file.extract_led_times(),
        'condition': opto_file['opto_conditions'],
        'level': opto_file['opto_levels']
    })
    optotagging_table = optotagging_table.sort_values(by='Start', axis=0)
    

    stop_times = []
    names = []
    descriptions = []
    for ii, row in optagging_table.iterrows():
        condition = known_conditions[row["condition"]]
        stop_times.append(row["start_time"] + condition.duration)
        names.append(condition.name)
        descriptions.append(condition.description)

    optotagging_table["stop_time"] = stop_times
    optotagging_table["name"] = names
    optotagging_table["description"] = descriptions
    optotagging_table["duration"] = optotagging_table["stop_time"] - optotagging_table["start_time"]
    optotagging_table.drop(columns="condition", inplace=True)

    optotagging_table.to_csv(args['output_opto_table_path'], index=False)
    return {'output_opto_table_path': args['output_opto_table_path']}


def main():

    mod = ArgSchemaParserPlus(schema_type=InputParameters, output_schema_type=OutputParameters)
    output = build_opto_table(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))

    
if __name__ == "__main__":
    main()

